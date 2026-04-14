from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from loguru import logger

from backend.src.domain.schemas.doc import Doc, DocMetadata, Section
from backend.src.services.processing.utils.loader import text_mimes
from backend.src.settings.fosra_paths import fosra_paths
from backend.src.storage.utils.converters import DomainStruct, ulid_factory

_MARKDOWN_ANCHOR_RE = re.compile(r"\s*\{#[^}]+\}\s*$")
_CALLOUT_HEADING_RE = re.compile(r"^(:::|!!!).*")

# ─── heading depth patterns for PDF sections (flat heading_path from docling) ──
# numeric:   "2.3.1 Heading" → depth 3, "2 Heading" → depth 1
_NUMERIC_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\b")
# roman numeral prefix: "I. Intro", "VI. Results"
_ROMAN_PREFIX_RE = re.compile(
    r"^(I{1,3}|IV|V|VI{0,3}|IX|X{1,2}(?:I{0,3}|IV|V|VI{0,3})?)\s*\.\s+", re.IGNORECASE
)
# letter appendix prefix: "A. Foo"
_LETTER_APPENDIX_RE = re.compile(r"^[A-Z]\. ")
_YAML_FRONTMATTER_RE = re.compile(r"^---\n[\s\S]*?\n---\n+", re.MULTILINE)

# heading path extraction params — tuned via BO (100-trial GPSampler)
# merge_min: merge sections below N chars forward into the next section
# split_max: split sections above N chars by distributing chunks evenly
_DEFAULT_MERGE_MIN = 380
_DEFAULT_SPLIT_MAX = 3500

# ─── markdown-aware heading parser ─────────────────────────────────────────────

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"!?\[(.*?)\]\([^)]*\)")

# ─── qdrant nav breadcrumb (top of file) ──────────────────────────────────────
# pattern 1: "* [Documentation](url)\n  * [Page Name]" — category index pages
_MD_NAV_CATEGORY_RE = re.compile(
    r"^\* \[Documentation\][^\n]*\n\s*\* [^\[\n]+(?:$|\n)", re.MULTILINE
)
# pattern 2: "* [Documentation](url)\n  * [Section](url)\n  * [Page Name]\n" — regular pages
_MD_NAV_RE = re.compile(
    r"^\* \[Documentation\][^\n]*\n\s*\* \[[^\]]+\][^\n]*\n\s*\* [^\[\n]+\n+",
    re.MULTILINE,
)

# ─── footer patterns ──────────────────────────────────────────────────────────
_SPHINX_FOOTER_RE = re.compile(
    r"\n\nCreated using \[Sphinx\].*?Documentation last generated:.*?(?=\n\n|\Z)",
    re.DOTALL,
)
_SQLA_SEARCH_RE = re.compile(r"\*\*Search terms:\*\*[\s\S]*?(?=\n\n[A-Z#]|$)")
_SQLA_BREADCRUMB_RE = re.compile(r"\[Contents[\s\S]*?\| \[Table[\s\S]*?\n\n")

# ─── mid-document patterns ─────────────────────────────────────────────────────
_SPHINX_PARA_LINK_RE = re.compile(r"\[¶\]\([^)]+\)")
_DOCUS_CALLOUT_RE = re.compile(r"^> \[![A-Z]+\]\s*$", re.MULTILINE)
_MKDOCS_COLON_RE = re.compile(r"^:::(?:\w+)?\n[\s\S]*?^:::", re.MULTILINE)
_MKDOCS_ADMONITION_RE = re.compile(r"^!!! \w+\s*$", re.MULTILINE)
_MKDOCS_CODE_CALLOUT_RE = re.compile(r"^\?\?\? \w+ ")
_MDX_IMPORT_RE = re.compile(r"^import \S+ from '@[\w/-]+';", re.MULTILINE)
_MDX_COMPONENT_RE = re.compile(r"^<(?:Intro|Pitfall|CAUTION|WARNING)>\n", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")

# ─── gutenberg ebook markers ───────────────────────────────────────────────────
# strip header: everything before *** START OF THE PROJECT GUTENBERG EBOOK
# strip footer: everything after *** END OF THE PROJECT GUTENBERG EBOOK
_GUTENBERG_START_RE = re.compile(
    r"^[\s\S]*?(\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*)",
    re.MULTILINE,
)
_GUTENBERG_END_RE = re.compile(
    r"(\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*)[\s\S]*$",
    re.MULTILINE,
)

# ─── plain-text chapter/section markers ───────────────────────────────────────
# roman numerals with titles: "I. The Adventures of Sherlock Holmes"
# require 2+ roman chars to avoid matching standalone letters like "D."
_ROMAN_NUMERAL_RE = re.compile(
    r"\n+([IVXLCDM]{2,}\.?\s+[A-Z][^\n]{0,60})", re.MULTILINE
)
# all-caps title lines (standalone, not license text): "THE REPAIRER OF REPUTATIONS"
# allow leading whitespace and variable blank lines before the title
# allow punctuation (periods, commas) in titles like "CASTLE OF INDOLENCE."
_ALL_CAPS_TITLE_RE = re.compile(
    r"\n+[ \t]*([A-Z][A-Z\s.,-]{3,59})[ \t]*\n\n",
    re.MULTILINE,
)
# numbered sections: "1. Chapter One", "Story I", "Part One"
_NUMBERED_SECTION_RE = re.compile(
    r"\n\n((?:Chapter|Story|Part|Section|Book)\s+[IVXLCDM\d]+\.?\s+.+)",
    re.MULTILINE,
)
# "THE END", "POSTSCRIPT", "FINIS" — standalone end markers
# allow variable blank lines before the marker, optional period at end
_END_MARKER_RE = re.compile(
    r"\n+[ \t]*(THE END|POSTSCRIPT|FINIS)[ \t]*\.?\n\n",
    re.MULTILINE,
)


def _clean_md(text: str) -> str:
    """Apply all markdown cleaning patterns."""
    # strip frontmatter
    text = _YAML_FRONTMATTER_RE.sub("", text)
    # strip qdrant nav breadcrumbs
    text = _MD_NAV_CATEGORY_RE.sub("", text)
    text = _MD_NAV_RE.sub("", text)
    # strip sphinx paragraph links: [¶](#id)
    text = _SPHINX_PARA_LINK_RE.sub("", text)
    # strip sphinx-generated footer
    text = _SPHINX_FOOTER_RE.sub("", text)
    # strip sqlachemy search bar
    text = _SQLA_SEARCH_RE.sub("", text)
    # strip sqlalchemy breadcrumb line
    text = _SQLA_BREADCRUMB_RE.sub("", text)
    # strip docusaurus callouts: > [!NOTE]
    text = _DOCUS_CALLOUT_RE.sub("", text, count=1)
    # strip mkdocs colon blocks: :::python ... :::
    text = _MKDOCS_COLON_RE.sub("", text)
    # strip mkdocs admonitions: !!! note
    text = _MKDOCS_ADMONITION_RE.sub("", text, count=1)
    # strip mkdocs code callouts: ??? example "..."
    text = _MKDOCS_CODE_CALLOUT_RE.sub("", text)
    # strip mdx component tags: <Intro>
    text = _MDX_COMPONENT_RE.sub("", text)
    # strip mdx imports
    text = _MDX_IMPORT_RE.sub("", text, count=1)
    # strip html comments
    text = _HTML_COMMENT_RE.sub("", text)
    return text


def _clean_gutenberg_text(text: str) -> str:
    """Strip Project Gutenberg header/footer from plain text ebooks."""
    text = _GUTENBERG_START_RE.sub(r"\1", text)
    text = _GUTENBERG_END_RE.sub(r"\1", text)
    return text


def _infer_chapters(text: str) -> str:
    """Convert detected chapter/section markers to markdown headings.

    Detects all-caps titles, roman numerals, numbered sections, and end markers
    in plain text files and converts them to # headings for consistent parsing.
    """
    # roman numerals with titles
    text = _ROMAN_NUMERAL_RE.sub(r"\n\n# \1", text)
    # numbered sections
    text = _NUMBERED_SECTION_RE.sub(r"\n\n# \1", text)
    # all-caps titles (but not license text or short headers)
    # be conservative: only convert lines that look like chapter titles
    text = _ALL_CAPS_TITLE_RE.sub(r"\n\n# \1\n\n", text)
    # end markers
    text = _END_MARKER_RE.sub(r"\n\n# \1\n\n", text)
    return text


def _extract_md_sections_by_heading(
    file_path: Path,
    merge_min: int,
    split_max: int,
) -> list[Section]:
    """parse markdown file directly by heading boundaries.

    docling's HierarchicalChunker loses heading metadata for markdown because the
    md parser emits SectionHeaderItem with empty text and the actual heading lives
    in the following TextItem. We parse headings directly instead.
    """

    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = _clean_md(text)

    sections: list[Section] = []
    current_heading: str | None = None
    current_heading_path: list[str] = []
    current_texts: list[str] = []
    current_pages: set[int] = set()  # markdown has no pages
    existing_headings: list[str] = []  # tracks seen headings for deduplication

    def flush() -> Section | None:
        return _flush_section(
            current_heading,
            current_heading_path,
            current_texts,
            current_pages,
            len(sections),
        )

    # split at heading boundaries (#### or higher = level 4+ for subsections)
    parts = re.split(r"\n(?=#{1,6}\s+)", text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # check if this part starts with a heading
        heading_match = _MD_HEADING_RE.match(part)
        if heading_match:
            # flush previous section
            if current_texts:
                sec = flush()
                if sec:
                    sections.append(sec)

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            # strip markdown links first: [text](url) → text
            heading_text = _MD_LINK_RE.sub(r"\1", heading_text)
            # strip trailing anchor: "{#some-id}"
            heading_text = _MARKDOWN_ANCHOR_RE.sub("", heading_text).strip()
            # ignore empty headings
            if not heading_text:
                heading_text = None
            else:
                existing_headings.append(heading_text)

            if level == 1:
                current_heading_path = [heading_text] if heading_text else []
            elif heading_text:
                current_heading_path = current_heading_path[: level - 1] + [
                    heading_text
                ]

            current_heading = heading_text

            # content after the heading line
            content = part[heading_match.end() :].strip()
            if content:
                current_texts.append(content)
            else:
                current_texts = []
        else:
            # continuation of current section
            if current_texts:
                current_texts[-1] += "\n\n" + part
            elif sections:
                # orphan content before any heading → attach to last section
                sections[-1].section_text = (
                    (sections[-1].section_text or "") + "\n\n" + part
                )

    if current_texts:
        sec = flush()
        if sec:
            sections.append(sec)

    # apply merge only — splits are handled by the chunker
    if merge_min > 0:
        sections = _merge_small(sections, merge_min)

    for i, sec in enumerate(sections):
        sec.section_index = i

    return sections


def _extract_text_sections(
    file_path: Path,
    merge_min: int,
    split_max: int,
) -> list[Section]:
    """Parse plain text files by inferring chapter/section structure.

    Handles Project Gutenberg ebooks and other plain text narratives by:
    1. Stripping Gutenberg header/footer boilerplate
    2. Detecting chapter/section markers (all-caps titles, roman numerals, etc.)
    3. Converting markers to markdown headings
    4. Parsing using the heading-based markdown extractor
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    # clean gutenberg boilerplate first
    text = _clean_gutenberg_text(text)
    # infer chapter structure and convert to markdown headings
    text = _infer_chapters(text)
    # write to temp .md file so we can reuse the md parser
    # (or pass text directly via a modified version)
    # For simplicity, write temp file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write(text)
        temp_path = Path(f.name)

    try:
        sections = _extract_md_sections_by_heading(temp_path, merge_min, split_max)
    finally:
        temp_path.unlink(missing_ok=True)

    return sections


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


def _flush_section(
    heading: str | None,
    heading_path: list[str] | None,
    texts: list[str],
    pages: set[int] | None,
    index: int,
) -> Section | None:
    if not texts:
        return None
    combined_text = "\n\n".join(texts)
    if not combined_text.strip():
        return None
    section_pages = sorted(pages) if pages else [1]

    return Section(
        section_text=combined_text,
        heading=heading,
        heading_path=heading_path,
        start_page=section_pages[0],
        end_page=section_pages[-1],
        section_index=index,
    )


def _merge_small(sections: list[Section], min_chars: int) -> list[Section]:
    if not sections or min_chars <= 0:
        return sections
    merged: list[Section] = []
    pending: Section | None = None
    for sec in sections:
        if pending is not None:
            ptext: str = pending.section_text or ""
            stext: str = sec.section_text or ""
            combined_hp: list[str] | None = (
                sec.heading_path if sec.heading_path else pending.heading_path
            )

            combined = _flush_section(
                sec.heading if sec.heading else pending.heading,
                combined_hp,
                ptext.split("\n\n") + stext.split("\n\n"),
                {
                    pending.start_page or 1,
                    pending.end_page or 1,
                    sec.start_page or 1,
                    sec.end_page or 1,
                },
                pending.section_index,
            )
            if combined:
                ctext: str = combined.section_text or ""
                if len(ctext) < min_chars:
                    pending = _flush_section(
                        combined.heading,
                        combined.heading_path,
                        ctext.split("\n\n"),
                        {combined.start_page or 1, combined.end_page or 1},
                        combined.section_index,
                    )
                else:
                    merged.append(combined)
                    pending = None
            else:
                pending = None
        elif len(sec.section_text or "") < min_chars:
            pending = sec
        else:
            merged.append(sec)
    if pending is not None:
        if merged:
            last = merged[-1]
            merged_hp: list[str] = (last.heading_path or []) + (
                pending.heading_path or []
            )
            _replacement = _flush_section(
                last.heading,
                merged_hp,
                (last.section_text or "").split("\n\n")
                + (pending.section_text or "").split("\n\n"),
                {
                    last.start_page or 1,
                    last.end_page or 1,
                    pending.start_page or 1,
                    pending.end_page or 1,
                },
                last.section_index,
            )
            if _replacement is not None:
                merged[-1] = _replacement
        else:
            merged.append(pending)
    return merged


def _split_large(sections: list[Section], max_chars: int) -> list[Section]:
    if max_chars <= 0:
        return sections
    result: list[Section] = []
    for sec in sections:
        if len(sec.section_text or "") <= max_chars:
            result.append(sec)
            continue
        chunks_text = (sec.section_text or "").split("\n\n")
        current_texts: list[str] = []
        current_chars = 0
        idx = sec.section_index
        for chunk_t in chunks_text:
            if current_chars + len(chunk_t) > max_chars and current_texts:
                s = _flush_section(
                    sec.heading,
                    sec.heading_path,
                    current_texts,
                    {sec.start_page or 1},
                    idx,
                )
                if s:
                    result.append(s)
                    idx += 1
                current_texts = []
                current_chars = 0
            current_texts.append(chunk_t)
            current_chars += len(chunk_t)
        if current_texts:
            s = _flush_section(
                sec.heading,
                sec.heading_path,
                current_texts,
                {sec.end_page or 1},
                idx,
            )
            if s:
                result.append(s)
    return result


# ─── section hierarchy builder ─────────────────────────────────────────────


def _numeric_parent_prefix(prefix: str) -> str | None:
    """given '2.3.1' return '2.3'; given '2' return None."""
    parts = prefix.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


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
        depth_stack2: dict[
            int, Section
        ] = {}  # depth → most recent numerically-anchored section
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
            return _extract_md_sections_by_heading(file_path, merge_min, split_max)

        # .txt → text-aware parser with chapter inference
        if suffix == ".txt":
            return _extract_text_sections(file_path, merge_min, split_max)

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
