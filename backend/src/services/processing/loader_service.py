from __future__ import annotations

import asyncio
import mimetypes
import pprint
from pathlib import Path
from pprint import pp
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from loguru import logger
from ulid import ULID

from backend.src.domain.enums import VectorStoreType
from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    UserPreferences,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import (
    Doc,
    DocMetadata,
    MDNFile,
    PDFMetadata,
    TextMetadata,
)
from backend.src.services.conversation import llm_service
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.vector_service import VectorService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


import markitdown
from rich.traceback import install

# _ = install(show_locals=True, word_wrap=True)


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


# NOTE: issues that remain: hash not implemented, other pdf parser options not available, pdf metadata type is limited to PyMuPDFParser
class LoaderService:
    # parse files from blobs from browser, minimal changes could allow for path parsing
    @staticmethod
    def parse_files(files: list[str | MDNFile]) -> list[Doc]:
        from content_types import EXTENSION_TO_CONTENT_TYPE
        from content_types import get_content_type as get_mime
        from langchain_community.document_loaders.parsers import PyMuPDFParser
        from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
        from langchain_community.document_loaders.parsers.txt import TextParser
        from langchain_core.document_loaders import Blob

        from backend.src.services.processing.utils.loader import code_mimes

        docs = []

        for file in files:

            if isinstance(file, str):
                # determine route to parse based on file type
                mime_type = get_mime(file)

                print(mime_type)
            elif isinstance(file, MDNFile):
                # NOTE: This is for browser received files as bytes, Will have to switch to magic library (magic.from_buffer) when i switch to browser ingest
                raise NotImplementedError("MDNFiles cant be ingested yet")
            else:
                raise RuntimeError("Incorrect File Type at Ingestion")

            id = ulid_factory()

            match mime_type:
                # NOTE: deal with PDF specifics later

                case "text/markdown" | "text/plain":
                    print("entered text")
                    file_bytes = to_bytes(file)

                    # TODO: hash bytes

                    blob: Blob = Blob.from_data(
                        file_bytes,
                        mime_type=mime_type,
                        path=file,
                    )

                    print(blob)

                    text_docs: list[Document] = TextParser().parse(blob)

                    for lc_doc in text_docs:

                        d: Doc = Doc(
                            id=ulid_factory(),
                            page_content=lc_doc.page_content,
                            metadata=DocMetadata(mime_type=mime_type, source=file),
                        )

                        docs.append(d)

                case _ if mime_type in code_mimes:
                    "code in code mimes"
                    file_bytes = to_bytes(file)

                    blob: Blob = Blob.from_data(
                        file_bytes,
                        mime_type=mime_type,
                        path=file,
                    )

                    text_docs: list[Document] = TextParser().parse(blob)

                    for lc_doc in text_docs:
                        #!todo: combine all documents and use page content to get doc hash

                        d: Doc = Doc(
                            id=ulid_factory(),
                            page_content=lc_doc.page_content,
                            metadata=DocMetadata(mime_type=mime_type, source=file),
                        )

                        docs.append(d)
                # case "application/pdf":
                #     # pdf parsing
                #     file_bytes = to_bytes(file)
                #
                #     blob: Blob = Blob.from_data(
                #         file_bytes,
                #         mime_type=mime_type,
                #         path=file,
                #     )
                #
                #     print(blob)
                #
                #     pdf_docs: list[Document] = PyMuPDFParser(mode="single").parse(
                #         blob=blob
                #     )
                #
                #     for lc_doc in pdf_docs:
                #         lc_doc.metadata["mime_type"] = mime_type
                #
                #         d.id = id
                #         docs.append(d)

                case _:
                    pass

        return docs


async def _chunk(result):
    chunks = await ChunkerService().chunk_documents(docs=result, config=ChunkerConfig())

    return chunks

    # return chunks


if __name__ == "__main__":
    pass
