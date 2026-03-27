from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from backend.src.settings import VectorStoreConfig
from backend.src.services.retrieval.vector_service import VectorService

from .broker import broker


@broker.task
async def store_file_vectors(
    config: VectorStoreConfig,
    embed_config,
    chunks: list[Any],
) -> list[str]:
    """Upsert source documents and their vectors into the store."""

    try:
        logger.info(f"Starting vector upsert for {len(chunks)} chunks")
        upserted = await VectorService().upsert(
            config=config, embed_config=embed_config, chunks=chunks
        )

        if upserted is None:
            logger.warning("Vector upsert returned None")
            return []

        logger.info(f"Completed vector upsert for {len(upserted)} points")
        return [str(p.id) for p in upserted]

    except Exception as e:
        logger.error(f"Vector upsert failed: {e}")
        raise
