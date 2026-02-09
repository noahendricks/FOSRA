from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from langchain_unstructured import UnstructuredLoader
from llama_parse import ResultType
from loguru import logger
from ulid import ULID
from markitdown import MarkItDown


from backend.src.domain.exceptions import (
    APIKeyMissingError,
    ParserDependencyError,
    InvalidDocumentError,
    ParsingOperationError,
)
from backend.src.domain.schemas import (
    FileContent,
    ParserConfig,
    SourceFull,
)


from backend.src.domain.enums import OriginType, ParserType, DocumentType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# =============================================================================
# Base Parser Interface
# =============================================================================


class BaseParser(ABC):
    _registry: ClassVar[dict[str, type[BaseParser]]] = {}

    def __init__(self, session: AsyncSession | None = None):
        self.session = session

    @classmethod
    def register(cls, parser_type: ParserType):
        def decorator(parser_cls: type[BaseParser]):
            cls._registry[parser_type.value] = parser_cls
            return parser_cls

        return decorator

    @classmethod
    def get_parser(
        cls,
        parser_type: ParserType,
        session: AsyncSession | None = None,
    ) -> BaseParser | None:
        parser_cls = cls._registry.get(parser_type.value)

        if parser_cls:
            return parser_cls(session)

        logger.warning(f"No parser found for type: {parser_type.value}")
        return None

    @classmethod
    def get_available_parsers(cls) -> list[str]:
        return list(cls._registry.keys())

    parser_type: ClassVar[ParserType]
    display_name: ClassVar[str]
    supported_types: ClassVar[list[DocumentType]]

    @abstractmethod
    async def parse(
        self, file_content: FileContent, config: ParserConfig
    ) -> SourceFull:
        pass

    def supports_document_type(self, doc_type: DocumentType) -> bool:
        return doc_type in self.supported_types


# =============================================================================
# Local Parser Base Class
# =============================================================================


class LocalParser(BaseParser):
    @abstractmethod
    async def _initialize_parser(self, config: ParserConfig) -> None:
        pass


# =============================================================================
# API Parser Base Class
# =============================================================================


class APIParser(BaseParser):
    def _validate_api_key(self, config: ParserConfig) -> str:
        if not config.api_key:
            raise APIKeyMissingError(embedder_type=self.parser_type.value)

        return str(config.api_key)


# =============================================================================
# Unstructured Implementation
# =============================================================================
@BaseParser.register(ParserType.UNSTRUCTURED)
class UnstructuredParser(LocalParser):
    # FIX: Replace as primary with textract; make unstructured an option
    parser_type: ClassVar[ParserType] = ParserType.UNSTRUCTURED
    display_name: ClassVar[str] = "Unstructured"
    supported_types: ClassVar[list[DocumentType]] = [
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.DOC,
        DocumentType.PPTX,
        DocumentType.XLSX,
        DocumentType.HTML,
        DocumentType.TEXT,
        DocumentType.MARKDOWN,
    ]

    async def _initialize_parser(self, config: ParserConfig) -> None:
        pass

    async def parse(
        self, file_content: FileContent, config: ParserConfig
    ) -> SourceFull:
        start_time = time.time()
        file_path = file_content.file_path
        content_bytes = file_content.content

        try:
            print(
                "DEBUGPRINT[79]: parsers.py:211 (before loader = UnstructuredLoader()"
            )
            loader = UnstructuredLoader(
                file=io.BytesIO(content_bytes),
                post_processors=[],
                languages=[config.language],
                include_orig_elements=False,
                include_metadata=False,
                strategy="auto",
                metadata_filename=file_content.file_name,
            )

            docs = await loader.aload()

            content = "\n\n".join([doc.page_content for doc in docs])

            page_numbers = set()

            for element in docs:
                if hasattr(element, "metadata") and element.metadata:
                    page_num = element.metadata.get("page_number")
                    if page_num is not None:
                        page_numbers.add(page_num)

            page_count = (
                len(page_numbers) if page_numbers else max(1, len(content) // 2000)
            )

            parse_time = (time.time() - start_time) * 1000

            return SourceFull(
                source_id=str(ULID()),
                origin_type=OriginType.FILESYSTEM,
                origin_path=file_content.file_path,
                name=file_content.file_name,
                hash=file_content.file_hash,
                document_type=file_content.metadata.document_type,
                content=content,
                metadata=(file_content.metadata.to_dict()),
            )

        except ImportError as e:
            raise ParserDependencyError(
                parser_type=self.parser_type.value,
                dependency="unstructured",
                remediation="Install unstructured: pip install unstructured langchain-community",
            ) from e
        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.error(f"Unstructured parsing failed for {file_path}: {e}")
            raise ParsingOperationError(
                parser_type=self.parser_type.value,
                file_path=file_path,
                reason=str(e),
                remediation="Ensure the file is a valid document supported by Unstructured.",
            )


# =============================================================================
# Markdown Implementation
# =============================================================================
@BaseParser.register(ParserType.MARKDOWN)
class MarkdownParser(LocalParser):
    async def parse(
        self, file_content: FileContent, config: ParserConfig
    ) -> SourceFull:
        try:
            logger.info("entered parse")

            md = MarkItDown()

            if isinstance(file_content.content, bytes):
                raise ValueError(
                    "File content is bytes at parse time. should be string"
                )

            from markdown import markdown

            content = markdown(file_content.content)

            # NOTE: Creation of SourceID HERE
            return SourceFull(
                source_id=str(ULID()),
                origin_type=file_content.metadata.origin_type,
                content=content,
                metadata={"parser_used": self.parser_type.value},
                hash=file_content.file_hash,
                origin_path=file_content.file_path,
            )

        except UnicodeDecodeError as e:
            raise InvalidDocumentError(
                file_path=file_content.file_path,
                reason=f"File encoding error: {e}",
                remediation="Ensure file is UTF-8 encoded text",
            )
        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.error(f"Markdown parsing failed for {file_content.file_path}: {e}")
            raise ParsingOperationError(
                parser_type=self.parser_type.value,
                file_path=file_content.file_path,
                reason=str(e),
                remediation="Ensure the file is a valid markdown or text document.",
            )


# =============================================================================
# PyPDF Implementation
# =============================================================================


@BaseParser.register(ParserType.PYPDF)
class PyPDFParser(LocalParser):
    parser_type: ClassVar[ParserType] = ParserType.PYPDF
    display_name: ClassVar[str] = "PyPDF"
    supported_types: ClassVar[list[DocumentType]] = [DocumentType.PDF]

    async def _initialize_parser(self, config: ParserConfig) -> None:
        """No initialization needed."""
        pass

    async def parse(
        self, file_content: FileContent, config: ParserConfig
    ) -> SourceFull:
        """Parse PDF using pypdf."""
        start_time = time.time()

        try:
            import pypdf

            pdf_reader = pypdf.PdfReader(io.BytesIO(file_content.content))
            page_count = len(pdf_reader.pages)

            pages_to_read = page_count
            if config.max_pages:
                pages_to_read = min(page_count, config.max_pages)
                logger.debug(
                    f"Limiting to {pages_to_read}/{page_count} pages "
                    f"per config.max_pages"
                )

            content_parts = []
            for i in range(pages_to_read):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if text:
                    content_parts.append(f"## Page {i + 1}\n\n{text}")

            content = "\n\n".join(content_parts)

            parse_time = (time.time() - start_time) * 1000

            return SourceFull(
                content=content,
                origin_path=file_content.file_path,
                metadata={
                    "total_pages": page_count,
                    "parsed_pages": pages_to_read,
                },
            )

        except ImportError as e:
            raise ParserDependencyError(
                parser_type=self.parser_type.value,
                dependency="pypdf",
                remediation="Install pypdf: pip install pypdf",
            ) from e
        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.error(f"PyPDF parsing failed for {file_content.file_path}: {e}")
            raise ParsingOperationError(
                parser_type="PyPDF",
                file_path=file_content.file_path,
                reason=f"{str(e)}",
            )


# =============================================================================
# LlamaParse Implementation
# =============================================================================


@BaseParser.register(ParserType.LLAMA_PARSE)
class LlamaParseParser(APIParser):
    parser_type: ClassVar[ParserType] = ParserType.LLAMA_PARSE
    display_name: ClassVar[str] = "LlamaParse"
    supported_types: ClassVar[list[DocumentType]] = [
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.PPTX,
        DocumentType.XLSX,
    ]

    async def parse(
        self, file_content: FileContent, config: ParserConfig
    ) -> SourceFull:
        start_time = time.time()

        try:
            from llama_parse import LlamaParse

            api_key = self._validate_api_key(config)

            parser = LlamaParse(
                api_key=api_key,
                result_type=ResultType.MD,
                num_workers=4,
                verbose=False,
            )

            logger.info(f"Sending {file_content.file_path} to LlamaParse API")

            documents = await parser.aload_data(io.BytesIO(file_content.content))

            content = "\n\n".join([doc.text for doc in documents])
            page_count = len(documents)

            parse_time = (time.time() - start_time) * 1000

            logger.info(
                f"LlamaParse processed {page_count} pages in {parse_time:.2f}ms"
            )

            return SourceFull(
                content=content,
                origin_path=file_content.file_path,
                metadata={"documents_count": len(documents)},
            )

        except ImportError as e:
            raise ParserDependencyError(
                parser_type=self.parser_type.value,
                dependency="llama-parse",
                remediation="Install llama-parse: pip install llama-parse",
            ) from e
        except APIKeyMissingError:
            raise
        except InvalidDocumentError:
            raise
        except Exception as e:
            logger.error(f"LlamaParse failed for {file_content.file_path}: {e}")
            raise ParsingOperationError(
                parser_type="LlamaParse",
                file_path=file_content.file_path,
                reason=f"{str(e)}",
            )
