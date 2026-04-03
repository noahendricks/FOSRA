from __future__ import annotations

import asyncio
import time
from asyncio.tasks import Task
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from backend.src.domain.enums import ChunkerType
from backend.src.settings import CodeChunkerConfig
from backend.src.domain.schemas.doc import (
    Chunk,
    ChunkMetadata,
    Doc,
    DocMetadata,
    PDFMetadata,
    TextMetadata,
)
from backend.src.services.processing.hi_chunk import HiChunk, HiChunkStructurer
from backend.src.storage.models import ulid_factory

if TYPE_CHECKING:
    from backend.src.settings import ChunkerConfig

from uuid import uuid4

from chonkie import Visualizer, chunker

viz = Visualizer()

from backend.src.settings import ChunkerConfig

# !hack : to be implemented:  user config retrieval (env , database or in-memory)

# !todo: Fix class state references


class ChunkerService:
    @staticmethod
    async def chunk_documents(
        docs: list[Doc], config: ChunkerConfig
    ) -> list[list[Chunk]]:
        """Chunk documents based on file type and user preferences."""

        start_time = time.time()

        logger.info("Starting chunking of {} documents", len(docs))

        tasks: list[Task[list[Chunk]]] = []

        async with asyncio.TaskGroup() as group:
            for doc in docs:
                # !hack: NOT MAINTAINING TEXT LOCATION IN END CHUNKS - NEED OPTION THAT DOES
                doc_type = "PDF" if doc.is_pdf else "code" if doc.is_code else "text"

                logger.debug("Processing {} document: {}", doc_type, doc.id)
                try:
                    if doc.is_code:
                        ccode_task = group.create_task(
                            ChunkerService._chunk_code(
                                doc,
                                config,
                            )
                        )

                        tasks.append(ccode_task)

                    elif doc.is_text:
                        c_task = group.create_task(
                            ChunkerService._chunk_text(doc, config)
                        )

                        tasks.append(c_task)

                    # elif doc.is_pdf:
                    #     # if doc is pdf, chunk / (leave as) as pages unless page length > 300 tokens, then chunk w/ text chunker
                    #
                    #     cpdf_task = group.create_task(
                    #         ChunkerService._chunk_pdf(doc, config)
                    #     )
                    #
                    #     tasks.append(cpdf_task)
                except:
                    raise

        chunks: list[list[Chunk]] = []

        for t in tasks:
            chunks.append(t.result())

        logger.info("Completed chunking {} documents", len(docs))

        return chunks

    @staticmethod
    async def _chunk_code(doc: Doc, config: ChunkerConfig):
        """Handle code file chunking."""
        # use code chunker as default, respect user preference
        from code_chunker import ChunkerConfig as CodeChunkerConfig
        from code_chunker import CodeChunker

        chunker = CodeChunker(
            config=CodeChunkerConfig(
                include_imports=True,
                include_comments=True,
            )
        )

        from backend.src.services.processing.utils.loader import code_mimes

        parse_result = chunker.parse(
            doc.page_content, language=code_mimes[doc.metadata.mime_type]
        )

        parse_result.file_path = doc.metadata.source

        return [
            Chunk(
                text=cc.code,
                metadata=ChunkMetadata(
                    chunk_id=cc.name or str(uuid4()),
                    start_char=0,
                    end_char=0,
                    token_count=0,
                ),
            )
            for cc in parse_result.chunks
        ]

    @staticmethod
    async def _chunk_text(doc: Doc, config: ChunkerConfig) -> list[Chunk]:
        """Handle text file chunking.

        Uses pre-extracted sections from kreuzberg if available, otherwise
        falls back to flat HiChunk on the full page_content.
        """
        if doc.metadata.sections:
            return await ChunkerService._chunk_by_sections(doc, config)

        structurer = HiChunkStructurer(config=config)
        chunks = HiChunk.index(document=doc, structurer=structurer)
        return chunks

    @staticmethod
    async def _chunk_by_sections(doc: Doc, config: ChunkerConfig) -> list[Chunk]:
        """Chunk using pre-grouped sections (from docling or kreuzberg).

        Each section's text is passed to HiChunk independently, and every
        resulting chunk inherits the section's positional metadata.
        Uses section.section_text when available (docling), otherwise falls
        back to SectionGrouper.section_text() concatenation (kreuzberg).
        """
        from backend.src.services.processing.kreuzberg_extractor import (
            SectionGrouper,
        )

        all_chunks: list[Chunk] = []
        structurer = HiChunkStructurer(config=config)

        for section in doc.metadata.sections:
            section_text = (
                section.section_text
                if section.section_text
                else SectionGrouper.section_text(section)
            )
            if not section_text.strip():
                continue

            section_doc = Doc(
                id=doc.id,
                page_content=section_text,
                metadata=DocMetadata(
                    source=doc.metadata.source,
                    mime_type=doc.metadata.mime_type,
                    doc_id=doc.metadata.doc_id,
                    doc_title=doc.metadata.doc_title,
                    path=doc.metadata.path,
                    language=doc.metadata.language,
                    repo=doc.metadata.repo,
                    source_type=doc.metadata.source_type,
                    checksum=doc.metadata.checksum,
                    section_heading=section.heading,
                ),
            )

            section_chunks = HiChunk.index(document=section_doc, structurer=structurer)

            for c in section_chunks:
                c.metadata.page_number = section.start_page
                c.metadata.doc_title = doc.metadata.doc_title
                c.metadata.section_heading = section.heading
                c.metadata.element_ids = section.element_ids

            all_chunks.extend(section_chunks)

        return all_chunks

    # NOTE: ADD PDF CHUNKING LATER
    #       needs page level chunking,
    #       page number attribution,
    #       and metadata attribution
