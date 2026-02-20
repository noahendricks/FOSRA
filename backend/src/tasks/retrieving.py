from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from backend.src.domain.schemas import RerankResult, RerankerConfig
from backend.src.domain.schemas.source_schemas import (
    RetrievedResult,
)
from backend.src.services.retrieval.reranker_service import RerankerService

from .broker import broker


@broker.task
async def rerank_documents(
    query: str,
    documents: list[RetrievedResult],
    config: RerankerConfig,
    session: AsyncSession,
) -> RerankResult:
    """Rerank"""

    # NOTE: Currently single query
    
    try:
        reranked = await RerankerService().rerank_documents(
            query=query,
            documents=documents,
            config=config,
            session=session,
        )
        return reranked

    except Exception as e:
        logger.error(f"Reranking Task failed:{e}")
        raise
