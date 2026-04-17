from __future__ import annotations

import re
import tempfile
from pathlib import Path

from backend.src.domain.schemas.doc import Section

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


def _numeric_parent_prefix(prefix: str) -> str | None:
    """given '2.3.1' return '2.3'; given '2' return None."""
    parts = prefix.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def _clean_heading(heading: str | None) -> str | None:
    if not heading:
        return None
    heading = heading.strip()
    if _CALLOUT_HEADING_RE.match(heading):
        return None
    heading = _MARKDOWN_ANCHOR_RE.sub("", heading)
    return heading.strip() or None


# ─── section flush/merge/split logic ───────────────────────────────────────────────────────


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


# ─── section extraction ───────────────────────────────────────────────────────


def extract_md_sections_by_heading(
    file_path: Path,
    merge_min: int,
    split_max: int,
) -> list[Section]:
    """parse markdown file directly by heading boundaries."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = _clean_md(text)

    sections: list[Section] = []
    current_heading: str | None = None
    current_heading_path: list[str] = []
    current_texts: list[str] = []
    current_pages: set[int] = set()

    def flush() -> Section | None:
        return _flush_section(
            current_heading,
            current_heading_path,
            current_texts,
            current_pages,
            len(sections),
        )

    parts = re.split(r"\n(?=#{1,6}\s+)", text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        heading_match = _MD_HEADING_RE.match(part)
        if heading_match:
            if current_texts:
                sec = flush()
                if sec:
                    sections.append(sec)

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_text = _MD_LINK_RE.sub(r"\1", heading_text)
            heading_text = _MARKDOWN_ANCHOR_RE.sub("", heading_text).strip()
            if not heading_text:
                heading_text = None

            if level == 1:
                current_heading_path = [heading_text] if heading_text else []
            elif heading_text:
                current_heading_path = current_heading_path[: level - 1] + [
                    heading_text
                ]

            current_heading = heading_text

            content = part[heading_match.end() :].strip()
            if content:
                current_texts.append(content)
            else:
                current_texts = []
        else:
            if current_texts:
                current_texts[-1] += "\n\n" + part
            elif sections:
                sections[-1].section_text = (
                    (sections[-1].section_text or "") + "\n\n" + part
                )

    if current_texts:
        sec = flush()
        if sec:
            sections.append(sec)

    if merge_min > 0:
        sections = _merge_small(sections, merge_min)

    for i, sec in enumerate(sections):
        sec.section_index = i

    return sections


def extract_text_sections(
    file_path: Path,
    merge_min: int,
    split_max: int,
) -> list[Section]:
    """parse plain text files by inferring chapter/section structure."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = _clean_gutenberg_text(text)
    text = _infer_chapters(text)

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write(text)
        temp_path = Path(f.name)

    try:
        sections = extract_md_sections_by_heading(temp_path, merge_min, split_max)
    finally:
        temp_path.unlink(missing_ok=True)

    return sections
