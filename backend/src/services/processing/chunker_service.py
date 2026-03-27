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

        logger.info(f"Starting chunking of {len(docs)} documents")

        tasks: list[Task[list[Chunk]]] = []

        async with asyncio.TaskGroup() as group:
            for doc in docs:
                # !hack: NOT MAINTAINING TEXT LOCATION IN END CHUNKS - NEED OPTION THAT DOES
                doc_type = "PDF" if doc.is_pdf else "code" if doc.is_code else "text"

                logger.debug(f"Processing {doc_type} document: {doc.id}")
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

        logger.info(f"Completed chunking {len(docs)} documents")

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
        """Handle text file chunking."""
        chunker_type = config.preferred_strategy

        structurer = HiChunkStructurer(config=ChunkerConfig())

        chunks = HiChunk.index(document=doc, structurer=structurer)

        return chunks

    # NOTE: ADD PDF CHUNKING LATER
    #       needs page level chunking,
    #       page number attribution,
    #       and metadata attribution
