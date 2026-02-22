from __future__ import annotations

from pprint import pp
from typing import TYPE_CHECKING

from langchain_core.documents.base import Blob
from ulid import ULID

from backend.src.domain.schemas.doc import MDNFile, Doc, PDFMetadata, TextMetadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


import markitdown


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


class LoaderService:
    # parse files from blobs from browser, minimal changes could allow for path parsing
    @staticmethod
    def parse_files(files: list[MDNFile]) -> list[Doc]:
        from langchain_core.document_loaders import BaseLoader, BaseBlobParser, Blob
        from langchain_community.document_loaders.parsers import PyMuPDFParser
        from langchain_community.document_loaders.parsers.txt import TextParser
        from langchain_community.document_loaders.parsers.language.language_parser import (
            LanguageParser,
        )

        docs = []

        for file in files:
            if not file.bytes:
                continue

            # determine route to parse based on file type
            media_type = file.media_type.lower()

            # TODO: currently not parsing by page, gotta figure out that and other pdf parsers
            id = ulid_factory()
            if media_type == "application/pdf":
                # pdf parsing
                blob = Blob.from_data(file.bytes, mime_type="application/pdf")
                # NOTE: will need to set filetype, name, and other from mdn file
                pdf_docs = PyMuPDFParser(mode="page").parse(blob=blob)
                pdf_full = PyMuPDFParser(mode="single").parse(blob=blob)

                # TODO: use pdf_full to get doc hash

                for lc_doc in pdf_docs:
                    lc_doc.metadata["content_type"] = "pdf"

                    # parse by page, return to chunker as page, pull together but keep page info
                    doc = Doc.from_lc(lc_doc)
                    doc.id = id
                    docs.append(doc)

            elif media_type.startswith("text/") or media_type in [
                "text/markdown",
                "text/plain",
            ]:
                # text/markdown parsing
                blob = Blob.from_data(file.bytes)
                # NOTE: will need to set filetype, name, and other from mdn file

                text_docs = TextParser().parse(blob=blob)

                for lc_doc in text_docs:
                    lc_doc.metadata["content_type"] = "text"

                    # TODO: combine all Documents and use page content to get doc hash
                    doc = Doc.from_lc(lc_doc)
                    doc.id = ulid_factory()
                    docs.append(doc)

            else:
                # code file parsing - determine language from extension
                language = LoaderService._get_language_from_filename(file.name)
                if language:
                    blob = Blob.from_data(file.bytes)
                    # let languageparser auto-detect from filename extension
                    # NOTE: will need to set filetype, name, and other from mdn file
                    parser = LanguageParser()
                    code_docs = parser.parse(blob)
                    # TODO: combine all docs and use page content to get doc hash
                    for lc_doc in code_docs:
                        doc = Doc.from_lc(lc_doc)
                        doc.id = ulid_factory()
                        docs.append(doc)

        return docs

    @staticmethod
    def _get_language_from_filename(filename: str) -> str | None:
        extension_map = {
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".cs": "csharp",
            ".cobol": "cobol",
            ".cpy": "cobol",
            ".ex": "elixir",
            ".exs": "elixir",
            ".go": "go",
            ".java": "java",
            ".js": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".jsx": "javascript",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".lua": "lua",
            ".pl": "perl",
            ".pm": "perl",
            ".py": "python",
            ".rb": "ruby",
            ".rs": "rust",
            ".scala": "scala",
            ".sc": "scala",
            ".sql": "sql",
            ".ts": "typescript",
            ".tsx": "typescript",
        }

        import os

        _, ext = os.path.splitext(filename.lower())
        return extension_map.get(ext)


if __name__ == "__main__":
    pdf_bytes = to_bytes(
        "/home/roccoluxe/Documents/docs/09-frontend-ui/tsform/overview.md"
    )

    pdf_blob = Blob.from_data(data=pdf_bytes)

    mock_mdn = MDNFile(
        media_type="text/markdown",
        type=pdf_blob.mimetype or "",
        name=str(pdf_blob.path),
        size=0,
        bytes=pdf_bytes,
        webkit_relative_path=pdf_blob.source,
    )

    result = LoaderService.parse_files([mock_mdn])

    for p in result:
        if isinstance(p.metadata, TextMetadata):
            pp(p.model_dump())
