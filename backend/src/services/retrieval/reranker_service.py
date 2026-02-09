from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import logfire
from loguru import logger

from backend.src.domain.exceptions import (
    InvalidDocumentListError,
    RerankerAPIConfigError,
    RerankerNotFoundError,
    RerankingOperationError,
)
from backend.src.domain.schemas import (
    RerankResult,
    RerankerConfig,
    RetrievalConfig,
    RetrievedResult,
    VectorSearchResult,
)


from backend.src.domain.enums import FileType, RerankerType
from backend.src.services.retrieval.impls.rerankers import BaseReranker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RerankerService:
    _default_reranker = RerankerType.FLASHRANK

    @staticmethod
    def get_reranker(
        reranker_type: RerankerType,
        session: AsyncSession | None = None,
    ) -> BaseReranker:
        reranker = BaseReranker.get_reranker(reranker_type, session)

        if not reranker:
            available = BaseReranker.get_available_rerankers()
            raise RerankerNotFoundError(
                reranker_type=reranker_type.value,
                available_rerankers=available,
            )

        return reranker

    @staticmethod
    def get_available_rerankers() -> list[str]:
        available = BaseReranker.get_available_rerankers()
        logger.debug(f"Available rerankers: {available}")
        return available

    @staticmethod
    def _validate_config(config: RerankerConfig) -> None:
        if config.top_k < 1:
            raise RerankerAPIConfigError(
                user_id=config.user_id,
                config_id=config.config_id,
                reranker_type="unknown",
                reason=f"top_k must be >= 1, got {config.top_k}",
            )

        if config.score_threshold is not None:
            if not 0 <= config.score_threshold <= 1:
                raise RerankerAPIConfigError(
                    user_id="",
                    config_id=config.config_id,
                    reranker_type="unknown",
                    reason=f"score_threshold must be in [0, 1], got {config.score_threshold}",
                )

    @staticmethod
    async def rerank_documents(
        query: str,
        documents: list[RetrievedResult],
        config: RerankerConfig,
        session: AsyncSession,
        reranker_type: RerankerType | None = None,
        user_id: str | None = None,
    ) -> RerankResult:
        if not documents:
            logger.warning("No documents provided for reranking")

            raise InvalidDocumentListError(
                user_id=user_id,
                config_id=config.config_id,
                reason="Document list cannot be empty",
            )

        if not query or not query.strip():
            logger.error("Empty query provided")

            raise RerankerAPIConfigError(
                user_id=user_id or "",
                config_id=config.config_id,
                reranker_type=reranker_type.value if reranker_type else "unknown",
                reason="Query cannot be empty",
            )

        RerankerService._validate_config(config)

        reranker_type = reranker_type or RerankerService._default_reranker

        logger.info(
            f"Starting reranking: query='{query[:50]}...', "
            f"documents={len(documents)}, reranker={reranker_type.value}"
        )

        try:
            reranker: BaseReranker = RerankerService.get_reranker(
                reranker_type=reranker_type,
                session=session,
            )

            result = await reranker.rerank(
                query=query,
                documents=documents,
                config=config,
                user_id=user_id,
            )

            if result.errors:
                logger.warning(f"Reranking completed with errors: {result.errors}")
            else:
                logger.success(
                    f"Reranking complete: {result.filtered_count}/"
                    f"{result.original_count} documents, "
                    f"{result.rerank_time_ms:.2f}ms"
                )

            return result

        except (
            RerankerNotFoundError,
            RerankerAPIConfigError,
            InvalidDocumentListError,
        ):
            raise
        except Exception as e:
            logger.error(f"Reranking operation failed: {e}")
            raise RerankingOperationError(
                reranker_type=reranker_type.value,
                query=query,
                document_count=len(documents),
                reason=str(e),
            ) from e

    @staticmethod
    async def rerank_documents_batch(
        queries_and_documents: list[RetrievedResult],
        config: RerankerConfig,
        session: AsyncSession,
        reranker_type: RerankerType | None = None,
        user_id: str | None = None,
        max_concurrent: int = 3,
    ) -> list[RerankResult]:
        # FIX: NOT WORKING
        if not queries_and_documents:
            logger.warning("No query-document pairs provided for batch reranking")
            return []

        logger.info(
            f"Starting batch reranking: {len(queries_and_documents)} queries, "
            f"max_concurrent={max_concurrent}"
        )

        reranker_type = reranker_type or RerankerService._default_reranker
        _ = RerankerService.get_reranker(reranker_type, session)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def rerank_single_set(
            query: str,
            documents: list[RetrievedResult],
        ) -> RerankResult:
            """Rerank a single query-document set with concurrency control."""
            async with semaphore:
                try:
                    return await RerankerService.rerank_documents(
                        query=query,
                        documents=documents,
                        config=config,
                        session=session,
                        reranker_type=reranker_type,
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to rerank query '{query[:50]}...': {e}")
                    return RerankResult(
                        documents=documents,
                        reranker_used=reranker_type.value,  # type: ignore
                        original_count=len(documents),
                        errors=[str(e)],
                    )

        tasks = []
        for result in queries_and_documents:
            if not result:
                logger.warning("Skipping empty query-document pair in batch")

            for query, docs in result:
                tasks.append(
                    rerank_single_set(query=result.query_text, documents=result)
                )
        results = await asyncio.gather(
            *[
                rerank_single(query=query, documents=docs)
                for query, docs in queries_and_documents
            ],
            return_exceptions=False,
        )

        successful = sum(1 for r in results if not r.errors)
        logger.info(f"Batch reranking complete: {successful}/{len(results)} successful")

        return list(results)

    @staticmethod
    def filter_by_score(
        documents: list[dict[str, Any]],
        threshold: float,
        score_field: str = "score",
    ) -> list[dict[str, Any]]:
        if not 0 <= threshold <= 1:
            logger.warning(f"Invalid threshold {threshold}, using 0.0")
            threshold = 0.0

        filtered = [d for d in documents if d.get(score_field, 0) >= threshold]

        logger.debug(
            f"Filtered {len(documents) - len(filtered)} documents "
            f"below threshold {threshold}"
        )

        return filtered

    @staticmethod
    def apply_top_k(
        documents: list[dict[str, Any]],
        k: int,
        score_field: str = "score",
    ) -> list[dict[str, Any]]:
        if k < 1:
            logger.warning(f"Invalid k={k}, using k=1")
            k = 1

        limited = documents[:k]

        if len(documents) > k:
            logger.debug(
                f"Limited from {len(documents)} to {k} documents "
                f"(min_score={limited[-1].get(score_field, 'N/A')})"
            )

        return limited

    @staticmethod
    def sort_by_score(
        documents: list[dict[str, Any]],
        score_field: str = "score",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get(score_field, 0),
            reverse=descending,
        )

        if documents:
            top_score = sorted_docs[0].get(score_field, 0)
            bottom_score = sorted_docs[-1].get(score_field, 0)
            logger.debug(
                f"Sorted {len(documents)} documents by {score_field}: "
                f"range=[{bottom_score:.4f}, {top_score:.4f}]"
            )

        return sorted_docs
