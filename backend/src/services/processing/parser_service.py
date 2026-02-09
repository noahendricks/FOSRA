from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.src.domain.exceptions import (
    ParserNotFoundError,
    ParsingOperationError,
    ParserTimeoutError,
    UnsupportedDocumentTypeError,
)
from backend.src.domain.schemas import (
    FileContent,
    ParsedDocument,
    ParserConfig,
    SourceFull,
)


from backend.src.domain.enums import DocumentType, ParserType
from backend.src.services.processing.impls.parsers import BaseParser, MarkdownParser

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ParserService:
    _default_parser_order = [
        ParserType.MARKDOWN,
        ParserType.UNSTRUCTURED,
        ParserType.DOCLING,
        ParserType.PYPDF,
    ]

    @staticmethod
    def get_parser(
        parser_type: ParserType,
        session: AsyncSession,
    ) -> BaseParser:
        parser = BaseParser.get_parser(parser_type, session)

        if not parser:
            available = BaseParser.get_available_parsers()
            raise ParserNotFoundError(
                parser_type=parser_type.value,
                remediation=f"Available parsers: {', '.join(available)}",
            )

        return parser

    @staticmethod
    def detect_document_type(file_path: str) -> DocumentType:
        ext = Path(file_path).suffix.lower().lstrip(".")

        type_map = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.DOCX,
            "doc": DocumentType.DOC,
            "pptx": DocumentType.PPTX,
            "xlsx": DocumentType.XLSX,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
            "md": DocumentType.MARKDOWN,
            "markdown": DocumentType.MARKDOWN,
            "txt": DocumentType.TEXT,
            "text": DocumentType.TEXT,
        }

        detected = type_map.get(ext, DocumentType.UNKNOWN)
        logger.debug(f"Detected document type for {file_path}: {detected.value}")
        return detected

    @staticmethod
    def get_best_parser_for_document(
        file_content: FileContent,
        session: AsyncSession,
        preferred_parser: ParserType | None = None,
    ) -> BaseParser | None:
        doc_type = (
            file_content.metadata.document_type
            or ParserService.detect_document_type(file_path=file_content.file_path)
        )

        if preferred_parser:
            try:
                parser = ParserService.get_parser(preferred_parser, session)
                if parser.supports_document_type(doc_type):
                    logger.info(
                        f"Using preferred parser {preferred_parser.value} "
                        f"for {doc_type.value}"
                    )
                    return parser
                else:
                    logger.warning(
                        f"Preferred parser {preferred_parser.value} does not "
                        f"support {doc_type.value}"
                    )
            except ParserNotFoundError:
                logger.warning(f"Preferred parser {preferred_parser.value} not found")

        for parser_type in ParserService._default_parser_order:
            try:
                parser = ParserService.get_parser(parser_type, session)

                if parser.supports_document_type(doc_type):
                    logger.info(
                        f"Selected {parser_type.value} parser for {doc_type.value}"
                    )
                    return parser
            except ParserNotFoundError:
                continue

        logger.warning(f"No compatible parser found for {doc_type.value}")
        return None

    @staticmethod
    async def _execute_single_parse(
        parser: BaseParser,
        file_content: FileContent,
        config: ParserConfig,
    ) -> SourceFull:
        """Execute a single parse operation with timeout handling."""
        try:
            result: SourceFull = await asyncio.wait_for(
                parser.parse(file_content, config=config),
                timeout=config.timeout_seconds,
            )

            print(f"DEBUGPRINT[80]: parser_service.py:158: result={result}")

            return result

        except asyncio.TimeoutError:
            raise ParserTimeoutError(
                parser_type=parser.parser_type.value,
                file_path=file_content.file_path,
                timeout_seconds=config.timeout_seconds,
            )
        except Exception as e:
            logger.error(
                f"Parser {parser.parser_type.value} failed for {file_content.file_path}: {e}"
            )
            raise ParsingOperationError(
                parser_type=parser.parser_type.value,
                file_path=file_content.file_path,
                reason=str(e),
            ) from e

    @staticmethod
    async def parse_document(
        file_content: FileContent,
        config: ParserConfig,
        session: AsyncSession,
        preferred_parser: ParserType | None = None,
    ) -> SourceFull:
        logger.info(f"Starting parse for document: {file_content.file_name}")

        parser = MarkdownParser(session=session)
        try:
            result = await ParserService._execute_single_parse(
                file_content=file_content,
                parser=parser,
                config=config,
            )

            return result

        except (ParserTimeoutError, ParsingOperationError) as e:
            logger.warning(
                f"Primary parser {parser.parser_type.value} failed: {e.message}"
            )

            if not config.fallback_parsers:
                raise

            primary_error = e
            result = None

        if config.fallback_parsers:
            logger.info(f"Attempting {len(config.fallback_parsers)} fallback parser(s)")

            for fallback_type in config.fallback_parsers:
                try:
                    fallback_parser = ParserService.get_parser(
                        parser_type=fallback_type,
                        session=session,
                    )

                    if fallback_parser.parser_type == parser.parser_type:
                        logger.debug(
                            f"Skipping fallback {fallback_type.value} (same as primary)"
                        )
                        continue

                    doc_type = ParserService.detect_document_type(
                        file_content.file_path
                    )

                    if not fallback_parser.supports_document_type(doc_type):
                        logger.debug(
                            f"Skipping fallback {fallback_type.value} "
                            f"(doesn't support {doc_type.value})"
                        )
                        continue

                    logger.info(f"Trying fallback parser: {fallback_type.value}")

                    fallback_result = await ParserService._execute_single_parse(
                        parser=fallback_parser,
                        file_content=file_content,
                        config=config,
                    )

                    if result is None or not result.content:
                        result = fallback_result

                except ParserNotFoundError:
                    logger.warning(f"Fallback parser {fallback_type.value} not found")
                    continue
                except (ParserTimeoutError, ParsingOperationError) as e:
                    logger.warning(
                        f"Fallback parser {fallback_type.value} failed: {e.message}"
                    )
                    continue

        if result:
            logger.warning(
                f"All parsers completed with errors for {file_content.file_path}, "
                "returning best attempt"
            )
            return result

        logger.error(f"All parsing attempts failed for {file_content.file_path}")
        raise primary_error  # type: ignore

    @staticmethod
    async def parse_documents(
        files: list[FileContent],
        config: ParserConfig,
        session_factory: async_sessionmaker[AsyncSession],
        preferred_parser: ParserType | None = None,
        max_concurrent: int = 3,
    ) -> list[SourceFull]:
        if not files:
            logger.warning("No documents provided for parsing")
            return []

        logger.info(
            f"Parsing {len(files)} documents with max_concurrent={max_concurrent}"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single(file_content: FileContent) -> SourceFull:
            """Parse a single document with concurrency control."""
            async with semaphore:
                async with session_factory() as session:
                    try:
                        return await ParserService.parse_document(
                            file_content=file_content,
                            config=config,
                            session=session,
                            preferred_parser=preferred_parser,
                        )
                    except Exception as e:
                        logger.error(f"Failed to parse {file_content.file_path}: {e}")
                        return SourceFull(
                            content="",
                            origin_type=file_content.metadata.origin_type,
                            hash=file_content.file_hash,
                            origin_path=file_content.file_path,
                        )

        results = await asyncio.gather(
            *[parse_single(path) for path in files],
            return_exceptions=False,
        )

        return results

    @staticmethod
    def get_supported_document_types(
        parser_type: ParserType,
        session: AsyncSession,
    ) -> list[DocumentType]:
        parser = ParserService.get_parser(parser_type, session)
        return parser.supported_types

    @staticmethod
    def get_parsers_for_document_type(
        document_type: DocumentType,
        session: AsyncSession,
    ) -> list[ParserType]:
        compatible_parsers = []

        for parser_type_str in ParserService.get_available_parsers():
            try:
                parser_type = ParserType(parser_type_str)
                parser = ParserService.get_parser(parser_type, session)

                if parser.supports_document_type(document_type):
                    compatible_parsers.append(parser_type)

            except (ValueError, ParserNotFoundError):
                continue

        logger.debug(
            f"Found {len(compatible_parsers)} parsers for {document_type.value}"
        )
        return compatible_parsers
