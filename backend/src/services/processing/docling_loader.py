from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from loguru import logger

from backend.src.domain.schemas.doc import Doc, DocMetadata, Section
from backend.src.storage.utils.converters import DomainStruct, ulid_factory

_MARKDOWN_ANCHOR_RE = re.compile(r"\s*\{#[^}]+\}\s*$")
_CALLOUT_HEADING_RE = re.compile(r"^(:::|!!!).*")
_YAML_FRONTMATTER_RE = re.compile(r"^---\n[\s\S]*?\n---\n\n?")
_YAML_KEY_VALUE_RE = re.compile(
    r"^[a-z_][a-z0-9_]*:(?:[ \t]+[^\n]*)?\n(?:[ \t]+[^\n]*\n)*(?:\n+)?"
)


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


def _clean_heading(heading: str | None) -> str | None:
    if not heading:
        return None
    heading = heading.strip()
    if _CALLOUT_HEADING_RE.match(heading):
        return None
    heading = _MARKDOWN_ANCHOR_RE.sub("", heading)
    return heading.strip() or None


class DoclingLoader:
    """Load and parse files using docling's structure-aware extraction.

    Uses docling's HierarchicalChunker to detect actual heading boundaries,
    producing sections with explicit heading hierarchy and page numbers.
    Each section's text is then sub-chunked by HiChunk for semantic granularity.
    """

    @staticmethod
    def parse_file_sync(file_path: str | Path, mime_type: str | None = None) -> Doc:
        """parse a single file with docling structure-aware extraction.

        Args:
            file_path: Path to the file
            mime_type: Optional mime type override

        Returns:
            doc with sections populated from docling hierarchical chunking
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if mime_type is None:
            mime = mimetypes.guess_type(str(path))[0]
            mime_type = mime if mime else "text/plain"

        sections = DoclingLoader._extract_sections(path)

        first_content = (
            sections[0].section_text
            if sections and sections[0].section_text
            else path.read_text(encoding="utf-8", errors="replace")
        )

        return Doc(
            id=ulid_factory(),
            page_content=first_content or "",
            metadata=DocMetadata(
                source=str(path.absolute()),
                mime_type=mime_type,
                doc_id=ulid_factory(),
                doc_title=path.name,
                sections=sections,
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
        dd
        suffix = file_path.suffix.lower()

        # .md and other text formats → markdown parser
        if mime in text_mimes or suffix == ".md":
            return _extract_md_sections_by_heading(file_path, merge_min, split_max)

        # .txt → text-aware parser with chapter inference
        if suffix == ".txt":
            return _extract_text_sections(file_path, merge_min, split_max)

        cache_dir = fosra_paths.data_dir / "docling"
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_dir = cache_dir

        Groups consecutive chunks with the same deepest heading into a section.
        Each section gets the full heading path, page range, and combined text.
        """
        try:
            converter = DocumentConverter()
            chunker = HierarchicalChunker()
            result = converter.convert(file_path)
            chunks = list(chunker.chunk(result.document))
        except Exception as ex:
            logger.warning("Docling failed for {}: {}", file_path, ex)
            raise DoclingParseError(file_path, str(ex), original_error=ex)

        if not chunks:
            raise DoclingParseError(file_path, "No content extracted from document")

        sections: list[Section] = []
        current_section_chunks: list[
            tuple[int, str, list[str]]
        ] = []  # (page_no, chunk_text, headings_path)
        current_heading: str | None = None
        current_heading_path: list[str] = []
        current_pages: set[int] = set()

        def _flush_section() -> Section | None:
            if not current_section_chunks:
                return None
            combined_text = "\n\n".join(text for _, text, _ in current_section_chunks)
            combined_text = _YAML_FRONTMATTER_RE.sub("", combined_text)
            while _YAML_KEY_VALUE_RE.match(combined_text):
                combined_text = _YAML_KEY_VALUE_RE.sub("", combined_text)
            section_pages = sorted(current_pages) if current_pages else [1]
            return Section(
                elements=[],
                section_text=combined_text,
                heading=current_heading,
                heading_path=current_heading_path if current_heading_path else None,
                start_page=section_pages[0],
                end_page=section_pages[-1],
                element_ids=[],
                section_index=len(sections),
            )

        for chunk in chunks:
            meta = chunk.meta
            headings: list[str] = [
                _clean_heading(h) or "" for h in getattr(meta, "headings", []) or []
            ]
            headings = [h for h in headings if h]
            raw_heading = headings[-1] if headings else None

            doc_items = getattr(meta, "doc_items", []) or []
            page_no: int | None = None
            if doc_items:
                prov = getattr(doc_items[0], "prov", []) or []
                if prov:
                    page_no = getattr(prov[0], "page_no", None)

            if raw_heading != current_heading and current_heading is not None:
                section = _flush_section()
                if section:
                    sections.append(section)
                current_section_chunks = []
                current_pages = set()

            current_heading = raw_heading
            current_heading_path = headings
            if page_no is not None:
                current_pages.add(page_no)
            current_section_chunks.append((page_no or 0, chunk.text, headings))

        section = _flush_section()
        if section:
            sections.append(section)

        sections = [s for s in sections if s.heading_path]

        return sections
