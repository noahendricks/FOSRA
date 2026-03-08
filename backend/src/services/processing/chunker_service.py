from __future__ import annotations

import asyncio
import time
from asyncio.tasks import Task
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from backend.src.domain.enums import ChunkerType
from backend.src.domain.schemas.doc import (
    Chunk,
    ChunkMetadata,
    Doc,
    PDFMetadata,
    TextMetadata,
)
from backend.src.storage.models import ulid_factory

if TYPE_CHECKING:
    from backend.src.domain.schemas.config import ChunkerConfig

from uuid import uuid4

from chonkie import (
    BaseChunker,
    CodeChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceTransformerEmbeddings,
    Visualizer,
    chunker,
)

viz = Visualizer()

from backend.src.domain.schemas.config import ChunkerConfig

# !hack : to be implemented:  user config retrieval (env , database or in-memory)

# !todo: Fix class state references
# pyright: reportIgnoreCommentWithoutRule=false


class ChunkerService:

    @staticmethod
    def _get_chunker(config: ChunkerConfig):
        """Get a chunker instance by type."""
        chunker_type = config.preferred_chunker_type

        chunker_config = config or ChunkerConfig()

        match chunker_type:
            case ChunkerType.LATE:
                from backend.src.domain.schemas.config import LateChunkerConfig

                late_config = LateChunkerConfig(
                    embedding_model=chunker_config.embedding_model,
                    chunk_size=chunker_config.chunk_size,
                )
                chunker = LateChunker(**late_config.model_dump())

                logger.info("initiated late chunker")

                return chunker

            case ChunkerType.RECURSIVE:

                from backend.src.domain.schemas.config import RecursiveChunkerConfig

                # !warn: config currently hardcoded, user config dump necessary for config

                recursive_config = RecursiveChunkerConfig(
                    chunk_size=chunker_config.chunk_size
                )

                chunker = RecursiveChunker(
                    chunk_size=recursive_config.chunk_size, rules=recursive_config.rules
                )

                logger.info("initiated recursive chunker")

                return chunker

            case ChunkerType.SEMANTIC:
                from backend.src.domain.schemas.config import SemanticChunkerConfig

                # !warn: config currently hardcoded, user config dump necessary for config

                semantic_config = SemanticChunkerConfig()

                embedding_model = SentenceTransformerEmbeddings(
                    semantic_config.embedding_model, trust_remote_code=True
                )

                chunker = SemanticChunker(
                    **{
                        **semantic_config.model_dump(),
                        "embedding_model": embedding_model,
                    }
                )

                logger.info(f"Initialized SEMANTIC chunker with model")

                return chunker

            case ChunkerType.NEURAL:

                from backend.src.domain.schemas.config import NeuralChunkerConfig

                neural_config = NeuralChunkerConfig(
                    min_characters_per_chunk=chunker_config.min_chunk_size
                )

                chunker = NeuralChunker(**neural_config.model_dump())

                logger.info(f"Initialized NEURAL chunker")

                return chunker

            case ChunkerType.CODE:
                from backend.src.domain.schemas.config import CodeChunkerConfig

                code_config = CodeChunkerConfig(chunk_size=chunker_config.chunk_size)

                chunker = CodeChunker(
                    **code_config.model_dump(),
                )

                logger.info(f"Initialized CODE chunker with chunk size")

                return chunker
            case _:
                pass

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
                    if doc.is_pdf:
                        # if doc is pdf, chunk / (leave as) as pages unless page length > 300 tokens, then chunk w/ text chunker
                        cpdf_task = group.create_task(
                            ChunkerService._chunk_pdf(doc, config)
                        )
                        tasks.append(cpdf_task)
                    elif doc.is_code:
                        ccode_task = group.create_task(
                            ChunkerService._chunk_code(doc, config)
                        )
                        tasks.append(ccode_task)
                    elif doc.is_text:
                        c_task = group.create_task(
                            ChunkerService._chunk_text(doc, config)
                        )
                        tasks.append(c_task)
                except:
                    raise

        chunks: list[list[Chunk]] = []

        for t in tasks:
            chunks.append(t.result())

        logger.info(f"Completed chunking {len(docs)} documents")

        return chunks

    @staticmethod
    async def _chunk_pdf(doc: Doc, config: ChunkerConfig) -> list[Chunk]:
        """Handle PDF chunking with page-based approach."""
        chunker_type = config.preferred_chunker_type if config else ChunkerType.SEMANTIC

        if not doc.is_pdf or not isinstance(doc.metadata, PDFMetadata):
            raise ValueError("PDF Doc Chunk Attempt Failed: Doc is not PDF")

        chunker = ChunkerService()._get_chunker(config)

        if chunker is None:
            raise RuntimeError("PDF Chunker from _get_chunker returned None")

        # token count approximation for 300 token threshold
        # FIX: NOT WORKING

        estimated_tokens = len(doc.page_content.split()) * 1.3  # rough conversion

        if estimated_tokens > 250:
            # chunk within the page using text chunker
            chunks = await ChunkerService()._chunk_with_chonkie(
                doc, chunker=chunker
            )  # pyright: ignore

            from pprint import pp
        else:
            # keep as single chunk (page)
            chunk_meta: ChunkMetadata = ChunkMetadata(
                chunk_id=str(uuid4()),
                doc_title=doc.metadata.source
                if doc.metadata.source
                else "Unknown File Name",
                doc_id=doc.id or "unknown",
                page_number=doc.metadata.page,
                token_count=int(estimated_tokens),
                start_index=0,
                end_index=len(doc.page_content),
            )

            chunk: Chunk = Chunk(text=doc.page_content, metadata=chunk_meta)

            chunks = [chunk]
            from pprint import pp

            for c in chunks:
                pp(
                    c,
                )
                print("\n \n")
        # FIX: END

        return chunks

    @staticmethod
    async def _chunk_code(doc: Doc, config: ChunkerConfig) -> list[Chunk]:
        """Handle code file chunking."""
        # use code chunker as default, respect user preference

        chunker = ChunkerService()._get_chunker(config)

        return await ChunkerService()._chunk_with_chonkie(
            doc, chunker=chunker  # pyright:ignore
        )

    @staticmethod
    async def _chunk_text(doc: Doc, config: ChunkerConfig) -> list[Chunk]:
        """Handle text file chunking."""
        chunker_type = config.preferred_chunker_type

        chunker: BaseChunker = ChunkerService()._get_chunker(config)  # pyright: ignore

        chunks = await ChunkerService()._chunk_with_chonkie(doc, chunker)

        return chunks

    @staticmethod
    async def _chunk_with_chonkie(doc: Doc, chunker: BaseChunker) -> list[Chunk]:
        """Use chonkie chunker to split text."""
        # async wrapper for chonkie chunker
        # WARN: leaving as sync now, async necessary in the future

        try:
            chonkie_chunks = chunker.chunk(doc.page_content)
            logger.info(f"Chunking with {chunker} ")

            viz.print(chonkie_chunks)

            chunks = []

            if isinstance(doc.metadata, PDFMetadata or TextMetadata):
                doc_title = (
                    doc.metadata.title if doc.metadata.title else "Unknown Doc Title"
                )
            else:
                doc_title = "Unknown Doc Title"

            for i, chonkie_chunk in enumerate(chonkie_chunks):
                chunk_meta = ChunkMetadata(
                    chunk_id=str(uuid4()),
                    doc_id=doc.id or "unknown",
                    doc_title=doc_title,
                    page_number=0,
                    token_count=getattr(chonkie_chunk, "token_count", None),
                    start_index=getattr(chonkie_chunk, "start_index", None),
                    end_index=getattr(chonkie_chunk, "end_index", None),
                )

                chunk = Chunk(text=chonkie_chunk.text, metadata=chunk_meta)

                if isinstance(doc.metadata, PDFMetadata):
                    chunk.metadata.page_number = doc.metadata.page

                chunks.append(chunk)

            return chunks

        except Exception as e:
            raise RuntimeError(f"Fatal Error While Chunking: {e}")
