from __future__ import annotations

import asyncio
import pprint
from pprint import pp
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from loguru import logger
from ulid import ULID

from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import (Doc, MDNFile, PDFMetadata, TextMetadata)
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.vector_service import VectorService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


from rich.traceback import install

_ = install(show_locals=True, word_wrap=True)

import markitdown


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


# NOTE: issues that remain: hash not implemented, other pdf parser options not available, pdf metadata type is limited to PyMuPDFParser
class LoaderService:
    # parse files from blobs from browser, minimal changes could allow for path parsing
    @staticmethod
    def parse_files(files: list[MDNFile]) -> list[Doc]:
        from langchain_community.document_loaders.parsers import PyMuPDFParser
        from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
        from langchain_community.document_loaders.parsers.txt import TextParser
        from langchain_core.document_loaders import Blob

        docs = []

        for file in files:
            if not file.bytes:
                continue

            # determine route to parse based on file type
            media_type = file.media_type.lower()

            # !todo : currently not parsing by page, gotta figure out that and other pdf parsers
            id = ulid_factory()

            # !note: will need to set filetype, name, and other from mdn file
            match media_type:
                case "application/pdf":
                    # pdf parsing
                    blob: Blob = Blob.from_data(file.bytes, mime_type="application/pdf")

                    pdf_docs: list[Document] = PyMuPDFParser(mode="single").parse(
                        blob=blob
                    )

                    # !todo : use pdf_full to get doc hash

                    for lc_doc in pdf_docs:
                        lc_doc.metadata["content_type"] = "pdf"

                        # parse by page, return to chunker as page, pull together but keep page info
                        d = Doc.from_lc(lc_doc)
                        d.id = id
                        docs.append(d)

                case "text/markdown" | "text/plain":
                    print("entered text")
                    # text/markdown parsing
                    blob: Blob = Blob.from_data(file.bytes, encoding="utf-8")

                    print("blob: \n", str(blob.data)[:100])

                    text_docs: list[Document] = TextParser().parse(blob)
                    print("text_docs: \n", text_docs)

                    for lc_doc in text_docs:
                        lc_doc.metadata["content_type"] = "text"

                        #!todo: combine all Documents and use page content to get doc hash
                        d: Doc = Doc.from_lc(lc_doc)
                        d.id = ulid_factory()
                        docs.append(d)

                case _:  # code file parsing - determine language from extension
                    language = LoaderService._get_language_from_filename(file.name)
                    if language:
                        blob = Blob.from_data(file.bytes)

                        parser = LanguageParser()

                        code_docs = parser.parse(blob)

                        # !todo : combine all docs and use page content to get doc hash
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


async def _chunk(result):
    chunks = await ChunkerService().chunk_documents(docs=result, config=ChunkerConfig())

    return chunks

    # return chunks


if __name__ == "__main__":
    md_bytes = to_bytes(
        "/home/roccoluxe/Documents/docs/09-frontend-ui/tsform/overview.md"
    )

    md_blob = Blob.from_data(data=md_bytes)

    pdf_bytes = to_bytes(
        "/home/roccoluxe/Documents/Misc/MakingMusic_DennisDeSantis.pdf"
    )

    pdf_blob = Blob.from_data(data=pdf_bytes)

    mock_mdn_pdf = MDNFile(
        media_type="application/pdf",
        type=pdf_blob.mimetype or "",
        name=str(pdf_blob.path),
        size=0,
        bytes=pdf_bytes,
        webkit_relative_path=pdf_blob.source,
    )

    mock_mdn_md = MDNFile(
        media_type="text/markdown",
        type=md_blob.mimetype or "",
        name=str(md_blob.path),
        size=0,
        bytes=md_blob.data,
        webkit_relative_path=md_blob.source,
    )

    result: list[Doc] = LoaderService.parse_files([mock_mdn_md])

    chunks = asyncio.run(
        ChunkerService().chunk_documents(docs=result, config=ChunkerConfig())
    )

    for c in chunks:
        print(c)

        print("\n \n")

    embedded_chunks = []

    for c in chunks:
        embedded_chunks.append(
            asyncio.run(
                EmbedderService().embed_chunks(chunks=c, config=EmbedderConfig())
            )
        )

    for e in embedded_chunks:

        logger.debug(f"storing {len(embedded_chunks)} vectorsj")

        ids = asyncio.run(
            VectorService().upsert(
                config=VectorStoreConfig(),
                chunks=[c for sub in embedded_chunks for c in sub],
            )
        )

        print(ids)

    # for p in result:
    #     if isinstance(p.metadata, TextMetadata):
    #         pp(p.model_dump())
    #     if isinstance(p.metadata, PDFMetadata):
    #         pp(p.model_dump())
