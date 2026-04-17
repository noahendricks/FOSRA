from __future__ import annotations

import mimetypes
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from hierarchical.postprocessor import ResultPostprocessor
from loguru import logger

from backend.src.domain.schemas.doc import Doc, DocMetadata, Section
from backend.src.services.processing.utils.docling_regex import (
    _NUMERIC_SECTION_RE,
    _clean_heading,
    _flush_section,
    _merge_small,
    _numeric_parent_prefix,
    extract_md_sections_by_heading,
    extract_text_sections,
)
from backend.src.services.processing.utils.loader import text_mimes
from backend.src.settings.fosra_paths import fosra_paths
from backend.src.storage.utils.converters import DomainStruct, ulid_factory

# heading path extraction params — tuned via BO (100-trial GPSampler)
# merge_min: merge sections below N chars forward into the next section
# split_max: split sections above N chars by distributing chunks evenly
_DEFAULT_MERGE_MIN = 380
_DEFAULT_SPLIT_MAX = 3500


class DoclingParseError(Exception):

    def __init__(
        self,
        file_path: str | Path,
        reason: str,
        original_error: Exception | None = None,
    ):
        self.file_path = str(file_path)
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Docling failed to parse {file_path}: {reason}")


class DoclingIngestionResult(DomainStruct, kw_only=True):
    file_path: str
    success: bool
    sections_count: int
    page_content_length: int
    error: str | None = None
    failed_pages: list[int] = []


# ─── section hierarchy builder ─────────────────────────────────────────────


def _build_section_tree(flat: list[Section], doc_id: str) -> list[Section]:
    """assign section_ids and wire parent/child relationships.

    strategy A (markdown / multi-level heading_paths):
        depth = len(heading_path); positional stack tracks current parent at each depth.

    strategy B (pdf / single-element heading_paths):
        numbered headings ('2.3.1 ...') → depth from dot count; parent resolved via prefix map.
        unnumbered headings → attached as children of the deepest numeric ancestor.

    returns only root sections; all children are accessible via section.children.
    """
    for i, sec in enumerate(flat):
        sec.section_id = f"{doc_id}:{i}"

    # choose strategy: if >50% of sections have multi-element heading_paths → markdown
    multi = sum(1 for s in flat if s.heading_path and len(s.heading_path) > 1)
    use_heading_path = multi > len(flat) * 0.5

    roots: list[Section] = []

    if use_heading_path:
        # STRATEGY A — markdown
        depth_stack: dict[int, Section] = {}
        for sec in flat:
            depth = len(sec.heading_path) if sec.heading_path else 1
            for d in list(depth_stack):
                if d >= depth:
                    del depth_stack[d]
            parent = depth_stack.get(depth - 1)
            if parent is not None:
                sec.parent_id = parent.section_id
                parent.children.append(sec)
            else:
                roots.append(sec)
            depth_stack[depth] = sec
    else:
        # STRATEGY B — pdf / ambiguous
        prefix_map: dict[str, Section] = {}  # "2.3" → section
        depth_stack2: dict[int, Section] = (
            {}
        )  # depth → most recent numerically-anchored section
        for sec in flat:
            heading = sec.heading or ""
            m = _NUMERIC_SECTION_RE.match(heading)
            if m:
                prefix = m.group(1)
                depth = prefix.count(".") + 1
                parent_prefix = _numeric_parent_prefix(prefix)
                parent = prefix_map.get(parent_prefix) if parent_prefix else None
                if parent is not None:
                    sec.parent_id = parent.section_id
                    parent.children.append(sec)
                else:
                    roots.append(sec)
                prefix_map[prefix] = sec
                for d in list(depth_stack2):
                    if d >= depth:
                        del depth_stack2[d]
                depth_stack2[depth] = sec
            else:
                # unnumbered — attach to deepest numerically-anchored ancestor
                if depth_stack2:
                    parent_sec = depth_stack2[max(depth_stack2)]
                    sec.parent_id = parent_sec.section_id
                    parent_sec.children.append(sec)
                else:
                    roots.append(sec)
                # unnumbered sections don't anchor subsequent siblings

    return roots


class DoclingLoader:
    """structure-aware document parsing using docling."""

    @staticmethod
    def parse_file_sync(file_path: str | Path, mime_type: str | None = None) -> Doc:
        """parse a single file with docling structure-aware extraction."""
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if mime_type is None:
            mime = mimetypes.guess_type(str(path))[0]
            mime_type = mime if mime else "text/plain"

        doc_id = ulid_factory()
        flat_sections = DoclingLoader._extract_sections(path)
        root_sections = _build_section_tree(flat_sections, doc_id)

        first_content = (
            root_sections[0].section_text
            if root_sections and root_sections[0].section_text
            else path.read_text(encoding="utf-8", errors="replace")
        )

        return Doc(
            id=ulid_factory(),
            page_content=first_content or "",
            metadata=DocMetadata(
                source=str(path.absolute()),
                mime_type=mime_type,
                doc_id=doc_id,
                doc_title=path.name,
                sections=root_sections,
            ),
        )

    @staticmethod
    def _extract_sections(
        file_path: Path,
        merge_min: int = _DEFAULT_MERGE_MIN,
        split_max: int = _DEFAULT_SPLIT_MAX,
    ) -> list[Section]:
        """extract sections by heading path. uses direct md parsing for .md files
        (docling chunker loses heading metadata for markdown), and HierarchicalChunker
        for all other formats. plain text files (.txt) use chapter-inference parsing."""

        mime = mimetypes.guess_type(str(file_path))[0] or ""
        suffix = file_path.suffix.lower()

        # .md and other text formats → markdown parser
        if mime in text_mimes or suffix == ".md":
            return extract_md_sections_by_heading(file_path, merge_min, split_max)

        # .txt → text-aware parser with chapter inference
        if suffix == ".txt":
            return extract_text_sections(file_path, merge_min, split_max)

        cache_dir = fosra_paths.data_dir / "docling"
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_dir = cache_dir

        try:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(do_ocr=False)
                    )
                }
            )
            result = converter.convert(file_path).document

            chunker = HierarchicalChunker()
            chunks = list(chunker.chunk(dl_doc=result))
        except Exception as ex:
            logger.warning("Docling failed for {}: {}", file_path, ex)
            raise DoclingParseError(file_path, str(ex), original_error=ex)

        if not chunks:
            raise DoclingParseError(file_path, "No content extracted from document")

        # group by full heading path tuple (works well for PDF/HTML)
        sections: list[Section] = []
        current_path_tuple: tuple[str, ...] | None = None
        current_heading: str | None = None
        current_heading_path: list[str] = []
        current_texts: list[str] = []
        current_pages: set[int] = set()

        for chunk in chunks:
            meta = chunk.meta
            raw_headings = [
                _clean_heading(h) or "" for h in getattr(meta, "headings", []) or []
            ]
            path_tuple = tuple(raw_headings)
            non_empty = [h for h in raw_headings if h]
            last_heading = non_empty[-1] if non_empty else None

            doc_items = getattr(meta, "doc_items", []) or []
            page_no: int | None = None
            if doc_items:
                prov = getattr(doc_items[0], "prov", []) or []
                if prov:
                    page_no = getattr(prov[0], "page_no", None)

            if path_tuple != current_path_tuple and current_texts:
                sec = _flush_section(
                    current_heading,
                    current_heading_path,
                    current_texts,
                    current_pages,
                    len(sections),
                )
                if sec:
                    sections.append(sec)
                current_texts = []
                current_pages = set()

            current_path_tuple = path_tuple
            current_heading = last_heading
            current_heading_path = non_empty
            if page_no is not None:
                current_pages.add(page_no)
            current_texts.append(chunk.text)

        sec = _flush_section(
            current_heading,
            current_heading_path,
            current_texts,
            current_pages,
            len(sections),
        )
        if sec:
            sections.append(sec)

        if merge_min > 0:
            sections = _merge_small(sections, merge_min)

        for i, sec in enumerate(sections):
            sec.section_index = i

        return sections
