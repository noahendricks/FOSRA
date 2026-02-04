from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from FOSRABack.src.domain.schemas import SourceFull, VectorStoreConfig
from FOSRABack.src.services.retrieval.vector_service import VectorService

from .broker import broker


@broker.task
async def store_file_vectors(
    sources: list[SourceFull],
    config: VectorStoreConfig,
    session_factory: async_sessionmaker,
) -> list[str]:
    """Upsert source documents and their vectors into the store."""

    try:
        logger.info(f"Starting vector upsert for {len(sources)} sources")
        upserted = await VectorService().upsert(
            sources=sources, config=config, session_factory=session_factory
        )

        logger.info(f"Completed vector upsert for {len(upserted)} sources")
        return upserted

    except Exception as e:
        logger.error(f"Vector upsert failed: {e}")
        raise
