from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.src.domain.exceptions import (
    EmbedderNotFoundError,
    EmbeddingOperationError,
)

from backend.src.domain.schemas import (
    ChunkFull,
    EmbedderConfig,
    EmbeddingResult,
    SourceFull,
)


from backend.src.domain.enums import EmbedderType
from backend.src.services.processing.impls.embedders import BaseEmbedder

if TYPE_CHECKING:
    pass


class EmbedderService:
    _semaphore = asyncio.Semaphore(3)

    @staticmethod
    def get_embedder(
        embedder_type: EmbedderType,
        session: AsyncSession,
    ) -> BaseEmbedder:
        embedder = BaseEmbedder.get_embedder(embedder_type, session)

        if not embedder:
            available = BaseEmbedder.get_available_embedders()

            raise EmbedderNotFoundError(
                embedder_type=embedder_type.value,
                remediation=f"Available embedders: {', '.join(available)}",
            )

        return embedder

    @staticmethod
    def get_available_embedders() -> list[str]:
        return BaseEmbedder.get_available_embedders()

    @staticmethod
    async def embed_texts(
        texts: list[str],
        config: EmbedderConfig,
        session: AsyncSession,
    ) -> EmbeddingResult:
        if not texts:
            logger.warning("No texts provided for embedding")
            return EmbeddingResult(
                embedder_used=config.embedder_type.value,
                dense_vectors=[],
            )

        try:
            async with EmbedderService._semaphore:
                embedder = EmbedderService.get_embedder(
                    embedder_type=config.embedder_type,
                    session=session,
                )

                result = await embedder.embed_texts(texts=texts, config=config)

                if result.errors:
                    logger.error(f"Embedding errors: {result.errors}")
                else:
                    logger.info(
                        f"Successfully embedded {len(texts)} texts in "
                        f"{result.embed_time_ms:.2f}ms"
                    )

                return result

        except EmbedderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise EmbeddingOperationError(
                operation="embed_texts",
                reason=str(e),
                remediation="Check embedder configuration and text input",
            ) from e

    @staticmethod
    async def embed_query(
        query: str,
        config: EmbedderConfig,
        session: AsyncSession,
    ) -> list[float]:
        if not query or not query.strip():
            raise EmbeddingOperationError(
                operation="embed_query",
                reason="Empty query provided",
                remediation="Provide a non-empty query string",
            )

        logger.debug(f"Embedding query with {config.embedder_type.value}")

        try:
            embedder = EmbedderService.get_embedder(
                embedder_type=config.embedder_type,
                session=session,
            )

            vector = await embedder.embed_query(query=query, config=config)

            logger.info(f"Successfully embedded query with {len(vector)} dimensions")

            return vector

        except EmbedderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise EmbeddingOperationError(
                operation="embed_query",
                reason=str(e),
                remediation="Check embedder configuration and query text",
            ) from e

    @staticmethod
    async def embed_chunks(
        chunks: list[ChunkFull],
        config: EmbedderConfig,
        session: AsyncSession,
    ) -> list[ChunkFull]:
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            logger.warning("No valid chunks with text to embed")
            return chunks

        logger.debug(
            f"Embedding {len(valid_chunks)}/{len(chunks)} valid chunks "
            f"with {config.embedder_type.value}"
        )

        try:
            embedder = EmbedderService.get_embedder(
                embedder_type=config.embedder_type,
                session=session,
            )

            embedded_chunks = await embedder.embed_chunks(
                chunks=chunks,
                config=config,
            )

            vectors_added = sum(
                1 for c in embedded_chunks if c.dense_vector is not None
            )
            logger.info(f"Successfully embedded {vectors_added} chunks")

            return embedded_chunks

        except EmbedderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"chunk embedding failed: {e}")
            raise EmbeddingOperationError(
                operation="embed_chunks",
                reason=str(e),
                remediation="check embedder configuration and chunk data",
            ) from e

    @staticmethod
    async def embed_source(
        source: SourceFull,
        config: EmbedderConfig,
        session: AsyncSession,
    ) -> SourceFull:
        if not source.chunks:
            logger.debug(f"source {source.source_id} has no chunks to embed")
            return source

        logger.debug(
            f"embedding source {source.source_id} with {len(source.chunks)} chunks"
        )

        try:
            embedder = EmbedderService.get_embedder(
                embedder_type=config.embedder_type,
                session=session,
            )

            embedded_source = await embedder.embed_source(
                source=source,
                config=config,
            )

            vectors_added = sum(
                1 for c in embedded_source.chunks if c.dense_vector is not None
            )

            logger.info(
                f"Successfully embedded source {source.source_id}: "
                f"{vectors_added}/{len(source.chunks)} chunks"
            )

            return embedded_source

        except EmbedderNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Source embedding failed for {source.source_id}: {e}")
            raise EmbeddingOperationError(
                operation="embed_source",
                reason=str(e),
                remediation="Check embedder configuration and source data",
            ) from e

    @staticmethod
    async def embed_sources(
        sources: list[SourceFull],
        config: EmbedderConfig,
        session_factory: async_sessionmaker[AsyncSession],
        max_concurrent: int = 3,
    ) -> list[SourceFull]:
        if not sources:
            logger.warning("No sources provided for embedding")
            return []

        sources_with_chunks = [s for s in sources if s.chunks]

        if not sources_with_chunks:
            logger.warning("No sources with chunks to embed")
            return sources

        total_chunks = sum(len(s.chunks) for s in sources_with_chunks)

        logger.debug(
            f"Embedding {len(sources_with_chunks)} sources with "
            f"{total_chunks} total chunks using {config.embedder_type.value}"
        )

        async with session_factory() as session:
            try:
                embedder = EmbedderService.get_embedder(
                    embedder_type=config.embedder_type,
                    session=session,
                )
            except EmbedderNotFoundError:
                logger.error(f"Embedder not found: {config.embedder_type.value}")
                return sources

            semaphore = asyncio.Semaphore(2)

            async def embed_single_source(source: SourceFull) -> SourceFull:
                async with semaphore:
                    if not source.chunks:
                        return source

                    try:
                        logger.info(f"Embedding source {source.source_id}...")
                        ss = await embedder.embed_source(
                            source=source,
                            config=config,
                        )
                        logger.info(f"Completed embedding source {source.source_id}.")
                        logger.info(f"Source Chunks: {len(ss.chunks)}")
                        return ss
                    except Exception as e:
                        logger.error(f"Failed to embed source {source.source_id}: {e}")
                        return source

            try:
                logger.info("Starting bulk source embedding...")
                tasks = [embed_single_source(source) for source in sources]
                results = await asyncio.gather(*tasks, return_exceptions=False)

                logger.info("Completed bulk source embedding.")
                total_embedded = sum(
                    sum(1 for c in s.chunks if c.dense_vector)
                    for s in results
                    if s.chunks
                )

                logger.info(
                    f"Successfully embedded {len(results)} sources: "
                    f"{total_embedded}/{total_chunks} total chunks"
                )

                return list(results)

            except Exception as e:
                logger.error(f"Bulk source embedding failed: {e}")
                return sources
