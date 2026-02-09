from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from FOSRABack.src.domain.schemas import (
    EmbedderConfig,
    RerankResult,
    RetrievedResult,
    VectorStoreConfig,
)
from FOSRABack.src.domain.schemas.config_schemas import RerankerConfig
from FOSRABack.src.services.retrieval.impls._utils import (
    deduplicate_results,
    transform_vector_results,
)
from FOSRABack.src.services.retrieval.vector_service import VectorService
from FOSRABack.src.tasks.processing import embed_query
from FOSRABack.src.tasks.retrieving import rerank_documents

from .broker import broker


async def search_online(
    query: str,
    documents: list[RetrievedResult],
) -> list[RetrievedResult] | None:
    """Search Online"""

    try:
        pass
    except Exception as e:
        logger.error(f"Online Search Task failed:{e}")
        raise


@broker.task
async def search_vector_store(
    query: str,
    vector_store_config: VectorStoreConfig,
    embedder_config: EmbedderConfig,
    reranker_config: RerankerConfig,
    session: AsyncSession,
) -> RerankResult:
    """Search Vector Store"""

    try:
        pass
        embedded_query = await embed_query(
            query=query, config=embedder_config, session=session
        )
        retrived_results: list[RetrievedResult] = await VectorService().search(
            query_text=query,
            query_vector=embedded_query,
            config=vector_store_config,
            session=session,
        )

        deduplicated_results: list[RetrievedResult] = deduplicate_results(
            results=retrived_results
        )

        reranked = await rerank_documents(
            query=query,
            documents=deduplicated_results,
            config=reranker_config,
            session=session,
        )

        return reranked
    except Exception as e:
        logger.error(f"Reranking Task failed:{e}")
        raise
