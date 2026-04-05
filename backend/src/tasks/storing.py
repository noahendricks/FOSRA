from typing import Any

from loguru import logger
from backend.src.settings import VectorStoreConfig
from backend.src.services.retrieval.vector_service import VectorService

from .broker import broker


@broker.task(max_execution_time=120)
async def store_file_vectors(
    config: VectorStoreConfig,
    embed_config,
    chunks: list[Any],
) -> list[str]:
    """Upsert source documents and their vectors into the store."""

    try:
        logger.info("Starting vector upsert for {} chunks", len(chunks))
        upserted = await VectorService().upsert(
            config=config, embed_config=embed_config, chunks=chunks
        )

        if upserted is None:
            logger.warning("Vector upsert returned None")
            return []

        logger.info("Completed vector upsert for {} points", len(upserted))
        return [str(p.id) for p in upserted]

    except Exception as e:
        logger.opt(exception=True).error("Vector upsert failed")
        raise
