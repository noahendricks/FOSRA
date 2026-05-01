from __future__ import annotations

import asyncio
import statistics
import time
from asyncio.tasks import Task
from typing import TYPE_CHECKING, cast

from loguru import logger

from backend.src.domain.schemas.doc import (
    Doc,
    HierarchicalChunk,
    Section,
    SectionMetadata,
    Subsection,
)

if TYPE_CHECKING:
    from backend.src.settings import ChunkerConfig

from uuid import uuid4


def count_tokens(text: str) -> int:
    # approximate: ~4 chars per token
    return len(text) // 4


def _ensure_section(section: Section | dict) -> Section:
    """Normalize a section to a Section object (handles dict from JSON deserialization)."""
    if isinstance(section, Section):
        # Recursively normalize children
        if section.children:
            normalized_children = [_ensure_section(c) for c in section.children]
            if any(isinstance(c, Section) for c in normalized_children):
                # Only create new Section if children were normalized
                return Section(
                    section_text=section.section_text,
                    heading=section.heading,
                    heading_path=section.heading_path,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    section_index=section.section_index,
                    section_id=section.section_id,
                    parent_id=section.parent_id,
                    children=normalized_children,
                )
        return section
    if isinstance(section, dict):
        # Recursively normalize children in dict
        children = section.get("children") or []
        normalized_children = [_ensure_section(c) for c in children]
        return Section(
            section_text=section.get("section_text"),
            heading=section.get("heading"),
            heading_path=section.get("heading_path"),
            start_page=section.get("start_page"),
            end_page=section.get("end_page"),
            section_index=section.get("section_index", 0),
            section_id=section.get("section_id"),
            parent_id=section.get("parent_id"),
            children=normalized_children,
        )
    return section  # type: ignore[return-value]


def _normalize_sections(sections: list[Section | dict]) -> list[Section]:
    """Normalize a list of sections, handling dicts from JSON deserialization."""
    return [_ensure_section(s) for s in sections]


def _is_badly_sectioned(sections: list[Section], threshold: float = 0.42) -> bool:
    """true if >threshold fraction of sections deviate strongly from the median token count.

    flat sections with no children (depth 1 across the board) where token counts vary
    wildly suggest the parser failed to extract meaningful structure.
    """
    if len(sections) < 4:
        return False
    counts = [count_tokens(s.section_text or "") for s in sections]
    if not any(counts):
        return True
    median = statistics.median(counts)
    if median == 0:
        return True
    outliers = sum(
        1 for c in counts if c > 3 * median or (median > 50 and c < median / 3)
    )
    return outliers / len(counts) > threshold


def _emit_section(
    sec: Section,
    doc_id: str,
    doc_title: str,
    max_tokens: int,
    result: list[Subsection],
) -> None:
    """emit sec and all its descendants into result as flat subsections."""
    text = sec.section_text or ""
    token_count = count_tokens(text)
    heading_level = len(sec.heading_path) if sec.heading_path else 1

    # emit section as-is if it has text — the section boundary IS the semantic unit.
    # the chunker is purely a fallback for badly-sectioned docs (no headings / flat structure).
    if text.strip():
        result.append(
            Subsection(
                text=text,
                metadata=SectionMetadata(
                    section_id=sec.section_id,
                    parent_id=sec.parent_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page_number=sec.start_page,
                    token_count=token_count,
                    section_heading=sec.heading,
                    heading_level=heading_level,
                    heading_path=sec.heading_path,
                ),
            )
        )

    # recurse into structural children regardless of whether parent had text
    for child in sec.children:
        _emit_section(child, doc_id, doc_title, max_tokens, result)


def _hi_chunks_to_sections(
    hi_chunks: list[HierarchicalChunk], doc_id: str
) -> list[Section]:
    """convert HierarchicalChunk tree to Section tree for badly-sectioned fallback."""
    roots: list[Section] = []
    section_idx = 0

    def convert(node: object, parent_id: str | None) -> Section:
        nonlocal section_idx
        sec = Section(
            section_text=getattr(node, "text", None),
            heading=None,
            heading_path=None,
            start_page=None,
            end_page=None,
            section_index=section_idx,
            section_id=f"{doc_id}:h{section_idx}",
            parent_id=parent_id,
        )
        section_idx += 1
        for child in getattr(node, "children", []):
            child_sec = convert(child, sec.section_id)
            sec.children.append(child_sec)
        return sec

    for node in hi_chunks:
        roots.append(convert(node, None))
    return roots


class ChunkerService:
    @staticmethod
    async def chunk_documents(
        docs: list[Doc], config: "ChunkerConfig"
    ) -> list[list[Subsection]]:
        """chunk documents based on file type and user preferences."""
        start_time = time.time()
        logger.info("Starting chunking of {} documents", len(docs))

        hi_structurer_ref: list[
            object
        ] = []  # lazy-init only if badly-sectioned doc found

        tasks: list[Task[list[Subsection]]] = []

        try:
            async with asyncio.TaskGroup() as group:
                for doc in docs:
                    doc_type = (
                        "PDF" if doc.is_pdf else "code" if doc.is_code else "text"
                    )
                    logger.debug("Processing {} document: {}", doc_type, doc.id)

                    if doc.is_code:
                        tasks.append(
                            group.create_task(ChunkerService._chunk_code(doc, config))
                        )
                    elif doc.is_text:
                        tasks.append(
                            group.create_task(
                                ChunkerService._chunk_structured(
                                    doc, config, hi_structurer_ref
                                )
                            )
                        )
                    elif doc.is_pdf:
                        tasks.append(
                            group.create_task(
                                ChunkerService._chunk_structured(
                                    doc, config, hi_structurer_ref
                                )
                            )
                        )
        except ExceptionGroup as eg:
            # Unpack ExceptionGroup to show the actual error
            for e in eg.exceptions:
                logger.error("Chunking task failed: {}: {}", type(e).__name__, e)
            raise eg.exceptions[0] from None

        chunks: list[list[Subsection]] = [t.result() for t in tasks]
        elapsed = time.time() - start_time
        logger.info("Completed chunking {} documents in {:.2f}s", len(docs), elapsed)
        return chunks

    @staticmethod
    async def _chunk_code(doc: Doc, config: "ChunkerConfig") -> list[Subsection]:
        """handle code file chunking via tree-sitter."""
        from backend.src.services.processing.code_ingest import extract_code_chunks
        from backend.src.services.processing.utils.parse_utils import code_mimes

        metadata = doc.metadata
        mime = (
            metadata.get("mime_type")
            if isinstance(metadata, dict)
            else getattr(metadata, "mime_type", None)
        )
        path = (
            metadata.get("path")
            if isinstance(metadata, dict)
            else getattr(metadata, "path", None)
        )
        doc_id = (
            metadata.get("doc_id")
            if isinstance(metadata, dict)
            else getattr(metadata, "doc_id", None)
        )
        doc_title = (
            metadata.get("doc_title")
            if isinstance(metadata, dict)
            else getattr(metadata, "doc_title", None)
        )

        language = code_mimes.get(mime, "python")
        return await extract_code_chunks(
            source_code=doc.page_content,
            file_path=path or doc.id,
            language=language,
            doc_id=doc_id or doc.id,
            doc_title=doc_title or path or doc.id,
        )

    @staticmethod
    async def _chunk_structured(
        doc: Doc,
        config: "ChunkerConfig",
        hi_structurer_ref: list[object],
    ) -> list[Subsection]:
        """emit subsections from the section tree; fall back to HiChunk for badly-sectioned docs."""
        sections = _normalize_sections(doc.metadata.sections)

        if not sections or _is_badly_sectioned(sections):
            logger.info(
                "badly-sectioned or empty sections for {}, falling back to HiChunk",
                getattr(getattr(doc, "metadata", None), "doc_title", "unknown"),
            )
            return await ChunkerService._chunk_hichunk_fallback(
                doc, config, hi_structurer_ref
            )

        return ChunkerService._sections_to_subsections(doc, sections, config)

    @staticmethod
    def _sections_to_subsections(
        doc: object,
        root_sections: list[Section],
        config: "ChunkerConfig",
    ) -> list[Subsection]:
        """walk the section tree and emit a flat list of subsections for upsert."""
        doc_id = getattr(getattr(doc, "metadata", None), "doc_id", None) or str(uuid4())
        doc_title = getattr(getattr(doc, "metadata", None), "doc_title", None) or ""
        result: list[Subsection] = []
        for root in root_sections:
            _emit_section(root, doc_id, doc_title, config.max_chunk_size, result)
        return result

    @staticmethod
    async def _chunk_hichunk_fallback(
        doc: Doc,
        config: "ChunkerConfig",
        hi_structurer_ref: list[object],
    ) -> list[Subsection]:
        """run HiChunk neural chunking and convert result to subsections."""
        from backend.src.services.processing.utils.hichunk import HiChunkStructurer

        if not hi_structurer_ref:
            hi_structurer_ref.append(HiChunkStructurer(config=config))
        structurer = cast("HiChunkStructurer", hi_structurer_ref[0])

        hi_chunks = await asyncio.to_thread(structurer.structure, doc)
        doc_id = doc.metadata.doc_id or str(uuid4())
        root_sections = _hi_chunks_to_sections(hi_chunks, doc_id)

        # reuse section-tree emission
        return ChunkerService._sections_to_subsections(doc, root_sections, config)

    # ── legacy fallback (PDF pages, no sections) ────────────────────────────────

    @staticmethod
    async def _chunk_pdf_pages(doc: Doc, config: "ChunkerConfig") -> list[Subsection]:
        """fallback page-level chunking for PDFs with no section structure."""
        from chonkie import TokenChunker

        page_marker = "\n__PDF_PAGE_BREAK__\n"
        pages = doc.page_content.split(page_marker)
        doc_id = doc.metadata.doc_id or str(uuid4())
        doc_title = doc.metadata.doc_title or ""

        all_subsections: list[Subsection] = []
        token_chunker = TokenChunker(chunk_size=300)

        for page_num, page_text in enumerate(pages, start=1):
            page_text = page_text.strip()
            if not page_text:
                continue

            page_tokens = len(page_text.split())
            if page_tokens > 300:
                for chunk in token_chunker.chunk(page_text):
                    page_id = f"{doc_id}:page{page_num}:{chunk.start_index}"
                    all_subsections.append(
                        Subsection(
                            text=chunk.text,
                            metadata=SectionMetadata(
                                section_id=page_id,
                                parent_id=None,
                                doc_id=doc_id,
                                doc_title=doc_title,
                                page_number=page_num,
                                token_count=chunk.token_count,
                                start_char=chunk.start_index,
                                end_char=chunk.end_index,
                            ),
                        )
                    )
            else:
                page_id = f"{doc_id}:page{page_num}"
                all_subsections.append(
                    Subsection(
                        text=page_text,
                        metadata=SectionMetadata(
                            section_id=page_id,
                            parent_id=None,
                            doc_id=doc_id,
                            doc_title=doc_title,
                            page_number=page_num,
                            token_count=page_tokens,
                        ),
                    )
                )

        return all_subsections
