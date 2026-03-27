from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from ulid import ULID

from backend.src.settings import ChunkerConfig
from backend.src.domain.schemas.doc import Doc, DocMetadata, MDNFile
from backend.src.services.processing.chunker_service import ChunkerService


def to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


from rich.traceback import install


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


from loguru import logger


# NOTE: issues that remain: hash not implemented, other pdf parser options not available, pdf metadata type is limited to PyMuPDFParser
class LoaderService:
    @staticmethod
    def _parse_files(files: list[str | Path | MDNFile]) -> list[Doc]:
        from content_types import EXTENSION_TO_CONTENT_TYPE
        from content_types import get_content_type as get_mime
        from langchain_community.document_loaders.parsers import PyMuPDFParser
        from langchain_community.document_loaders.parsers.language.language_parser import (
            LanguageParser,
        )
        from langchain_community.document_loaders.parsers.txt import TextParser
        from langchain_core.document_loaders import Blob

        from backend.src.services.processing.utils.loader import code_mimes

        # TODO: MAKE ASYNC
        docs = []

        for file in files:
            if isinstance(file, Path):
                file = file.as_posix()

            if isinstance(file, str):
                # determine route to parse based on file type
                mime_type = get_mime(file)

                logger.debug("Detected mime type: {}", mime_type)
            elif isinstance(file, MDNFile):
                # NOTE: This is for browser received files as bytes, Will have to switch to magic library (magic.from_buffer) when i switch to browser ingest
                raise NotImplementedError("MDNFiles cant be ingested yet")
            else:
                raise RuntimeError(f"Incorrect File Type at Ingestion: {type(file)}")

            id = ulid_factory()

            match mime_type:
                # NOTE: deal with PDF specifics later
                case "text/markdown" | "text/plain":
                    file_bytes = to_bytes(file)

                    # TODO: hash bytes

                    blob: Blob = Blob.from_data(
                        file_bytes,
                        mime_type=mime_type,
                        path=file,
                    )

                    text_docs: list[Document] = TextParser().parse(blob)

                    for lc_doc in text_docs:
                        d: Doc = Doc(
                            id=ulid_factory(),
                            page_content=lc_doc.page_content,
                            metadata=DocMetadata(
                                mime_type=mime_type,
                                source=file,
                                doc_id=ulid_factory(),
                                doc_title=Path(file).name,
                            ),
                        )

                        logger.debug(
                            "created {}: {}: {}",
                            d.metadata.mime_type,
                            d.metadata.source,
                            d.id,
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
                            metadata=DocMetadata(
                                mime_type=mime_type,
                                source=file,
                                doc_id=ulid_factory(),
                                doc_title=Path(file).name,
                            ),
                        )
                        logger.debug(
                            "created {}: {}: {}",
                            d.metadata.mime_type,
                            d.metadata.source,
                            d.id,
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

    @staticmethod
    def _parse_directory(dir_path: str | Path) -> list[Doc]:
        dir = Path(dir_path)
        file_limit = 100
        all_files_count = sum(1 for _ in dir.rglob("*") if _.is_file())

        files_list = []

        if all_files_count > file_limit:
            raise ValueError(
                f"Loader Error [Too Many Files]: The path [{dir_path}] has {all_files_count}, {all_files_count - file_limit} over the limit of {file_limit} "
            )

        else:
            for file_path in dir.rglob("*"):
                logger.debug("Found file: {}", file_path.as_posix())

                if file_path.is_file():
                    files_list.append(file_path)

            if files_list:
                logger.info("Running parse on {} files", len(files_list))
                all_files = LoaderService()._parse_files(files_list)
            else:
                all_files = []

            return all_files

    # public method
    @staticmethod
    def parse_user_paths(user_paths: list[str]):
        files_as_docs = []
        loose_files = []
        for path in user_paths:
            p = Path(path)
            if p.is_dir():
                logger.info("Processing directory: {}", p.as_posix())

                dir_files_list = LoaderService()._parse_directory(dir_path=p)

                files_as_docs.append(dir_files_list)
            if p.is_file():
                file_as_doc = loose_files.append(p)

        if loose_files:
            files_as_docs.append(LoaderService()._parse_files(loose_files))

        logger.debug("Returning {} docs", len(files_as_docs))
        return files_as_docs

    @staticmethod
    def parse_pdf():
        from docling.document_converter import DocumentConverter

        # NOTE: Will complete later, not important now
        source = ""
        converter = DocumentConverter()
        result = converter.convert(source)
        logger.debug("PDF parse result: {}", result.document.export_to_markdown())
