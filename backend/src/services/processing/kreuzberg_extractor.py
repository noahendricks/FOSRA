from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import Any

import kreuzberg
from kreuzberg import ExtractionConfig  # type: ignore[attr-defined]

from loguru import logger

from backend.src.domain.schemas.doc import Section


class OutputFormat(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    DJOT = "djot"
    HTML = "html"
    STRUCTURED = "structured"


class ResultFormat(str, Enum):
    ELEMENT_BASED = "element_based"
    OBJECT_BASED = "object_based"


class KreuzbergExtractor:
    @staticmethod
    async def extract_elements(
        file_path: str,
        output_format: OutputFormat = OutputFormat.PLAIN,
    ) -> list[dict[str, Any]]:
        config = ExtractionConfig(
            output_format=output_format,
            result_format=ResultFormat.ELEMENT_BASED,
        )

        result = await kreuzberg.extract_file(file_path, config=config)

        if not result.elements:
            logger.warning("Kreuzberg returned no elements for {}", file_path)
            return []

        return result.elements

    @staticmethod
    def extract_elements_sync(
        file_path: str,
        output_format: OutputFormat = OutputFormat.PLAIN,
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            KreuzbergExtractor.extract_elements(file_path, output_format)
        )


class SectionGrouper:
    """Split a flat element list into logical sections at heading boundaries."""

    HEADING_TYPES = {"title", "heading"}

    KNOWN_PAGE_HEADERS = {"hichunk", "a-rag", "agentic"}

    _NUMBERED_HEADING = __import__("re").compile(
        r"^(\d+(?:\.\d+)*\s+[A-Z][a-zA-Z]*(?:\s+[A-Za-z0-9:,.\-/]*)?)(?!\s*[\[\(←])", 0
    )
    _APPENDIX_HEADING = __import__("re").compile(
        r"^(Appendix\s+[A-Z](?:\.\d+)?|[A-Z]\.\d+)\b", 0
    )
    _KEYWORD_HEADING = __import__("re").compile(
        r"^(?:Abstract|Conclusion|Introduction|Methodology)\b", 0
    )

    @staticmethod
    def _is_clean_heading(line: str) -> bool:
        if "←" in line or "→" in line:
            return False
        if re.search(r"\[[A-Za-z0-9]+\s*[=:]", line):
            return False
        return True

    @staticmethod
    def _extract_heading_line(text: str) -> str | None:
        """Extract the most likely heading line from element text.

        Handles:
        1. "HiChunk\r\n2 Related Works\r\n..." -> "2 Related Works"
        2. "1 Introduction\r\n..." -> "1 Introduction"
        3. "Figure 1: ...\r\n2.1 Core Components" -> "2.1 Core Components"
        """
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        first = lines[0].strip().lower() if lines else ""

        if first in SectionGrouper.KNOWN_PAGE_HEADERS and len(lines) > 1:
            for line in lines[1:]:
                line = line.strip()
                if line and SectionGrouper._is_clean_heading(line):
                    if (
                        SectionGrouper._NUMBERED_HEADING.match(line)
                        or SectionGrouper._APPENDIX_HEADING.match(line)
                        or SectionGrouper._KEYWORD_HEADING.match(line)
                    ):
                        return line
            return lines[1].strip()

        for line in lines:
            line = line.strip()
            if line and SectionGrouper._is_clean_heading(line):
                if (
                    SectionGrouper._NUMBERED_HEADING.match(line)
                    or SectionGrouper._APPENDIX_HEADING.match(line)
                    or SectionGrouper._KEYWORD_HEADING.match(line)
                ):
                    return line

        return None

    @staticmethod
    def _is_section_heading(
        elem: dict[str, Any],
        prev_elem: dict[str, Any] | None = None,
        next_elem: dict[str, Any] | None = None,
    ) -> bool:
        etype = elem.get("element_type", "")
        if etype in SectionGrouper.HEADING_TYPES:
            return True
        text = elem.get("text", "").strip()
        if text.startswith("#"):
            return True

        if etype == "page_break":
            return False

        is_prev_page_break = (
            prev_elem is not None and prev_elem.get("element_type") == "page_break"
        )
        is_next_page_break = (
            next_elem is not None and next_elem.get("element_type") == "page_break"
        )

        if is_prev_page_break and is_next_page_break:
            cleaned = text.replace("\r\n", " ").replace("\n", " ").strip()
            words = cleaned.split()
            if words and words[0][0].isdigit() and len(cleaned) <= 3:
                return False
            return True

        if is_prev_page_break:
            heading_line = SectionGrouper._extract_heading_line(text)
            if heading_line and len(heading_line) >= 4:
                return True

        if etype == "narrative_text" or etype == "title":
            cleaned = text.replace("\r\n", " ").replace("\n", " ").strip()
            if 3 < len(cleaned) < 120:
                words = cleaned.split()
                if len(words) <= 6 and not cleaned.endswith((".", ":", ";")):
                    first_is_upper = cleaned[0].isupper() if cleaned else False
                    has_title_case = (
                        any(w[0].isupper() for w in words[1:])
                        if len(words) > 1
                        else False
                    )
                    if first_is_upper and (has_title_case or len(words) <= 3):
                        return True

            heading_line = SectionGrouper._extract_heading_line(text)
            if heading_line and len(heading_line) >= 4:
                return True

        return False

    @staticmethod
    def group(
        elements: list[dict[str, Any]],
        doc_title: str | None = None,
    ) -> list[Section]:
        """Group elements into sections split at heading/title boundaries.

        Sections are split at:
        - heading element types (title, heading)
        - markdown headings (# prefix)
        - page_break elements (ensures page-level attribution for PDFs)

        Args:
            elements: Flat list of kreuzberg element dicts
            doc_title: Optional document title for section metadata

        Returns:
            List of Section objects, each containing a list of elements and metadata
        """
        sections: list[Section] = []
        current_section: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_ids: list[str] = []

        for idx, elem in enumerate(elements):
            prev_elem = elements[idx - 1] if idx > 0 else None
            next_elem = elements[idx + 1] if idx < len(elements) - 1 else None
            etype = elem.get("element_type", "")
            is_heading = SectionGrouper._is_section_heading(elem, prev_elem, next_elem)
            is_page_break = etype == "page_break"

            if is_page_break and current_section:
                page_nums = [
                    e.get("metadata", {}).get("page_number")
                    for e in current_section
                    if e.get("element_type") != "page_break"
                    and e.get("metadata", {}).get("page_number") is not None
                ]
                start_page = min(page_nums) if page_nums else None
                end_page = max(page_nums) if page_nums else None

                sections.append(
                    Section(
                        elements=current_section,
                        heading=current_heading,
                        element_ids=current_ids,
                        start_page=start_page,
                        end_page=end_page,
                        section_index=len(sections),
                    )
                )
                current_section = []
                current_ids = []
                current_heading = None
                continue

            if is_heading:
                if current_section or current_heading:
                    page_nums = [
                        e.get("metadata", {}).get("page_number")
                        for e in current_section
                        if e.get("element_type") != "page_break"
                        and e.get("metadata", {}).get("page_number") is not None
                    ]
                    start_page = min(page_nums) if page_nums else None
                    end_page = max(page_nums) if page_nums else None

                    sections.append(
                        Section(
                            elements=current_section,
                            heading=current_heading,
                            element_ids=current_ids,
                            start_page=start_page,
                            end_page=end_page,
                            section_index=len(sections),
                        )
                    )
                    current_section = []
                    current_ids = []

                heading_text = elem.get("text", "").strip()
                extracted = SectionGrouper._extract_heading_line(heading_text)
                current_heading = extracted if extracted else heading_text[:80]
                current_section.append(elem)
                current_ids.append(elem.get("element_id", ""))
            else:
                current_section.append(elem)
                current_ids.append(elem.get("element_id", ""))

        if current_section:
            page_nums = [
                e.get("metadata", {}).get("page_number")
                for e in current_section
                if e.get("element_type") != "page_break"
                and e.get("metadata", {}).get("page_number") is not None
            ]
            start_page = min(page_nums) if page_nums else None
            end_page = max(page_nums) if page_nums else None

            sections.append(
                Section(
                    elements=current_section,
                    heading=current_heading,
                    element_ids=current_ids,
                    start_page=start_page,
                    end_page=end_page,
                    section_index=len(sections),
                )
            )

        return SectionGrouper._collapse_orphan_sections(sections)

    @staticmethod
    def _collapse_orphan_sections(sections: list[Section]) -> list[Section]:
        """Merge single-element sections with no heading into the next section.

        Pages like "13\\nProblem:..." get split into their own section, but the
        actual content (the problem/solution description) belongs to the next
        page. Collapse these orphans into the following section.
        """
        if len(sections) < 2:
            return sections

        collapsed: list[Section] = []
        i = 0
        while i < len(sections):
            sec = sections[i]
            is_orphan = (
                not sec.heading
                and len(sec.elements) == 1
                and sec.elements[0].get("element_type") != "page_break"
            )
            if is_orphan and i + 1 < len(sections):
                next_sec = sections[i + 1]
                merged_elements = sec.elements + next_sec.elements
                merged_ids = sec.element_ids + next_sec.element_ids
                merged_heading = next_sec.heading
                if not merged_heading and merged_elements:
                    first_text = (
                        merged_elements[0]
                        .get("text", "")
                        .replace("\r\n", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    if first_text:
                        merged_heading = first_text
                merged_start = sec.start_page
                merged_end = next_sec.end_page

                collapsed.append(
                    Section(
                        elements=merged_elements,
                        heading=merged_heading,
                        element_ids=merged_ids,
                        start_page=merged_start,
                        end_page=merged_end,
                        section_index=len(collapsed),
                    )
                )
                i += 2
            else:
                sec.section_index = len(collapsed)
                collapsed.append(sec)
                i += 1

        return collapsed

    @staticmethod
    def section_text(section: Section) -> str:
        """Concatenate element texts in a section."""
        return "\n".join(elem.get("text", "") for elem in section.elements)
