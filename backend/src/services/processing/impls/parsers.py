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
