from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

from backend.src.domain.schemas.doc import Doc, DocMetadata
from backend.src.services.processing.kreuzberg_extractor import (
    KreuzbergExtractor,
    SectionGrouper,
)
from backend.src.storage.utils.converters import ulid_factory


class KreuzbergLoader:
    """Load and parse files using kreuzberg element-based extraction."""

    @staticmethod
    async def parse_file(
        file_path: str | Path,
        mime_type: str | None = None,
    ) -> Doc:
        """Parse a single file with kreuzberg element-based extraction.

        Args:
            file_path: Path to the file
            mime_type: Optional mime type override

        Returns:
            Doc with sections populated from kreuzberg extraction
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if mime_type is None:
            mime = mimetypes.guess_type(str(path))[0]
            mime_type = mime if mime else "text/plain"

        elements = await KreuzbergExtractor.extract_elements(str(path))
        if not elements or not mime_type:
            content = path.read_text(encoding="utf-8", errors="replace")
            return Doc(
                id=ulid_factory(),
                page_content=content,
                metadata=DocMetadata(
                    source=str(path.absolute()),
                    mime_type=mime_type,
                    doc_id=ulid_factory(),
                    doc_title=path.name,
                ),
            )

        sections = SectionGrouper.group(elements, doc_title=path.name)

        first_page = (
            elements[0].get("metadata", {}).get("page_number", 1) if elements else 1
        )
        first_content = next(
            (
                e.get("text", "")
                for e in elements
                if e.get("element_type") != "page_break"
            ),
            "",
        )

        return Doc(
            id=ulid_factory(),
            page_content=first_content,
            metadata=DocMetadata(
                source=str(path.absolute()),
                mime_type=mime_type,
                doc_id=ulid_factory(),
                doc_title=path.name,
                sections=sections,
            ),
        )

    @staticmethod
    def parse_file_sync(file_path: str | Path, mime_type: str | None = None) -> Doc:
        """Synchronous wrapper for parse_file."""
        return asyncio.run(KreuzbergLoader.parse_file(file_path, mime_type))
