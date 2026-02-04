from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import logfire
from loguru import logger

from FOSRABack.src.domain.exceptions import (
    RerankerDependencyError,
    RerankerInitializationError,
    RerankingOperationError,
)
from FOSRABack.src.domain.exceptions import RerankerAPIConfigError
from FOSRABack.src.domain.schemas import (
    RerankResult,
    RerankerConfig,
    RetrievedResult,
)

from FOSRABack.src.domain.enums import RerankerType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# =============================================================================
# Base Reranker Interface
# =============================================================================


class BaseReranker(ABC):
    _registry: ClassVar[dict[str, type[BaseReranker]]] = {}

    reranker_type: ClassVar[RerankerType]
    display_name: ClassVar[str]

    def __init__(self, session: AsyncSession | None = None):
        self.session = session

    @classmethod
    def register(cls, reranker_type: RerankerType):
        def decorator(reranker_cls: type[BaseReranker]):
            cls._registry[reranker_type.value] = reranker_cls
            logger.debug(f"Registered reranker: {reranker_type.value}")
            return reranker_cls

        return decorator

    @classmethod
    def get_reranker(
        cls,
        reranker_type: RerankerType,
        session: AsyncSession | None = None,
    ) -> BaseReranker | None:
        reranker_cls = cls._registry.get(reranker_type.value)
        if reranker_cls:
            logger.debug(f"Creating reranker instance: {reranker_type.value}")
            return reranker_cls(session)

        logger.warning(f"Reranker not found in registry: {reranker_type.value}")
        return None

    @classmethod
    def get_available_rerankers(cls) -> list[str]:
        return list(cls._registry.keys())

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[RetrievedResult],
        config: RerankerConfig,
        user_id: str | None = None,
    ) -> RerankResult:
        pass


# =============================================================================
# Local Reranker Base Class
# =============================================================================


class LocalReranker(BaseReranker):
    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self._model = None
        self._initialized = False

    @abstractmethod
    async def _initialize_model(self) -> None:
        pass

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._initialize_model()
            self._initialized = True


# =============================================================================
# API Reranker Base Class
# =============================================================================


class APIReranker(BaseReranker):
    @abstractmethod
    async def _get_api_config(self, user_id: str) -> dict[str, Any]:
        pass


# =============================================================================
# Local Reranker Implementations
# =============================================================================


@BaseReranker.register(RerankerType.FLASHRANK)
class FlashRankReranker(LocalReranker):
    reranker_type = RerankerType.FLASHRANK
    display_name = "FlashRank"
    default_model = "ms-marco-TinyBERT-L-2-v2"

    @logfire.instrument("Initializing FlashRank model")
    async def _initialize_model(self) -> None:
        try:
            from flashrank import Ranker

            logger.debug(f"Loading FlashRank model: {self.default_model}")

            self._model = Ranker(model_name=self.default_model)

            logger.success(f"FlashRank model loaded: {self.default_model}")

        except ImportError as e:
            logger.error("FlashRank not installed")
            raise RerankerDependencyError(
                reranker_type=self.reranker_type.value,
                missing_dependency="flashrank",
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize FlashRank: {e}")
            raise RerankerInitializationError(
                reranker_type=self.reranker_type.value,
                reason=str(e),
            ) from e

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedResult],
        config: RerankerConfig,
        user_id: str | None = None,
    ) -> RerankResult:
        start_time = time.time()
        if not documents:
            logger.warning("No documents to rerank")
            return RerankResult(
                documents=[],
                reranker_used=self.reranker_type.value,
                original_count=0,
            )

        try:
            await self._ensure_initialized()

            logger.debug(
                f"Reranking {len(documents)} documents with FlashRank, "
                f"top_k={config.top_k}, threshold={config.score_threshold}"
            )

            from flashrank import RerankRequest

            passages = [
                {"id": d.chunk_id or i, "text": d.contents}
                for i, d in enumerate(documents)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)

            results: list[dict[str, Any]] = await asyncio.to_thread(
                self._model.rerank,  # type: ignore
                rerank_request,
            )

            reranked_docs: list[RetrievedResult] = []

            for res in results:
                original: RetrievedResult | None = next(
                    (d for d in documents if str(d.chunk_id) == res["id"]),
                    None,
                )
                if original:
                    doc = original
                    doc.result_rank = res["score"]
                    doc.similarity_score = res["score"]
                    reranked_docs.append(doc)

            if config.score_threshold is not None:
                before_filter = len(reranked_docs)
                reranked_docs = [
                    d
                    for d in reranked_docs
                    if d.reranker_score or 0 >= config.score_threshold
                ]
                logger.debug(
                    f"Filtered {before_filter - len(reranked_docs)} documents "
                    f"below threshold {config.score_threshold}"
                )

            reranked_docs = reranked_docs[: config.top_k]

            rerank_time = (time.time() - start_time) * 1000

            logger.info(
                f"FlashRank reranking complete: "
                f"{len(reranked_docs)}/{len(documents)} documents, "
                f"{rerank_time:.2f}ms"
            )

            return RerankResult(
                documents=reranked_docs,
                reranker_used=self.reranker_type.value,
                rerank_time_ms=rerank_time,
                original_count=len(documents),
                filtered_count=len(reranked_docs),
            )

        except (RerankerDependencyError, RerankerInitializationError):
            raise
        except Exception as e:
            logger.error(f"FlashRank reranking failed: {e}")
            raise RerankingOperationError(
                reranker_type=self.reranker_type.value,
                query=query,
                document_count=len(documents),
                reason=str(e),
            ) from e


@BaseReranker.register(RerankerType.CROSS_ENCODER)
class CrossEncoderReranker(LocalReranker):
    reranker_type = RerankerType.CROSS_ENCODER
    display_name = "Cross-Encoder"
    default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    async def _initialize_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            logger.debug(f"Loading cross-encoder model: {self.default_model}")
            self._model = CrossEncoder(self.default_model)
            logger.success(f"Cross-encoder model loaded: {self.default_model}")

        except ImportError as e:
            logger.error("sentence-transformers not installed")
            raise RerankerDependencyError(
                reranker_type=self.reranker_type.value,
                missing_dependency="sentence-transformers",
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            raise RerankerInitializationError(
                reranker_type=self.reranker_type.value,
                reason=str(e),
            ) from e

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedResult],
        config: RerankerConfig,
        user_id: str | None = None,
    ) -> RerankResult:
        start_time = time.time()

        if not documents:
            logger.warning("No documents to rerank")
            return RerankResult(
                documents=[],
                reranker_used=self.reranker_type.value,
                original_count=0,
            )

        try:
            await self._ensure_initialized()

            logger.debug(
                f"Reranking {len(documents)} documents with Cross-Encoder, "
                f"top_k={config.top_k}, threshold={config.score_threshold}"
            )

            pairs = [(query, d.contents or "") for d in documents]

            scores = await asyncio.to_thread(
                lambda: self._model.predict(pairs)  # type: ignore
            )

            for doc, score in zip(documents, scores):
                doc.reranker_score = float(score)
                doc.similarity_score = float(score)

            reranked_docs = sorted(
                documents,
                key=lambda x: x.reranker_score or 0,
                reverse=True,
            )

            if config.score_threshold is not None:
                before_filter = len(reranked_docs)
                reranked_docs = [
                    d
                    for d in reranked_docs
                    if d.reranker_score or 0 >= config.score_threshold
                ]
                logger.debug(
                    f"Filtered {before_filter - len(reranked_docs)} documents "
                    f"below threshold {config.score_threshold}"
                )

            reranked_docs = reranked_docs[: config.top_k]

            rerank_time = (time.time() - start_time) * 1000

            logger.info(
                f"Cross-encoder reranking complete: "
                f"{len(reranked_docs)}/{len(documents)} documents, "
                f"{rerank_time:.2f}ms"
            )

            return RerankResult(
                documents=reranked_docs,
                reranker_used=self.reranker_type.value,
                rerank_time_ms=rerank_time,
                original_count=len(documents),
                filtered_count=len(reranked_docs),
            )

        except (RerankerDependencyError, RerankerInitializationError):
            raise
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            raise RerankingOperationError(
                reranker_type=self.reranker_type.value,
                query=query,
                document_count=len(documents),
                reason=str(e),
            ) from e


# =============================================================================
# API Reranker Implementations
# =============================================================================


@BaseReranker.register(RerankerType.COHERE)
class CohereReranker(APIReranker):
    reranker_type = RerankerType.COHERE
    display_name = "Cohere"
    default_model = "rerank-english-v3.0"

    async def _get_api_config(self, user_id: str) -> dict[str, Any]:
        if not self.session:
            logger.error("No database session available for API config lookup")
            raise RerankerAPIConfigError(
                config_id="",
                reranker_type=self.reranker_type.value,
                user_id=user_id,
                reason="No database session",
            )

        try:
            from sqlalchemy import select

            from FOSRABack.src.storage.models import LLMConfigORM

            query = select(LLMConfigORM).filter(
                LLMConfigORM.user_id == user_id,
                LLMConfigORM.provider == "COHERE",
            )
            result = await self.session.execute(query)
            config = result.scalars().first()

            if not config or not config.api_key:
                logger.warning(f"Cohere API key not found for user {user_id}")
                raise RerankerAPIConfigError(
                    config_id="",
                    reranker_type=self.reranker_type.value,
                    user_id=user_id,
                    reason="API key not configured",
                )

            logger.debug(f"Found Cohere API config for user {user_id}")
            return {"api_key": config.api_key}

        except RerankerAPIConfigError:
            raise
        except Exception as e:
            logger.error(f"Failed to get Cohere API config: {e}")
            raise RerankerAPIConfigError(
                config_id="",
                reranker_type=self.reranker_type.value,
                user_id=user_id,
                reason=str(e),
            ) from e

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        config: RerankerConfig,
        user_id: str | None = None,
    ) -> RerankResult:
        start_time = time.time()

        if not documents:
            logger.warning("No documents to rerank")
            return RerankResult(
                documents=[],
                reranker_used=self.reranker_type.value,
                original_count=0,
            )

        if not user_id:
            logger.error("user_id required for Cohere reranker")
            raise RerankerAPIConfigError(
                config_id="",
                reranker_type=self.reranker_type.value,
                user_id="not provided",
                reason="required user_id not provided",
            )

        try:
            api_config = await self._get_api_config(user_id)

            logger.debug(
                f"Reranking {len(documents)} documents with Cohere, "
                f"top_k={config.top_k}, threshold={config.score_threshold}"
            )

            try:
                import cohere
            except ImportError as e:
                logger.error("cohere not installed")
                raise RerankerDependencyError(
                    reranker_type=self.reranker_type.value,
                    missing_dependency="cohere",
                ) from e

            client = cohere.AsyncClient(api_key=api_config["api_key"])

            texts = [d.get("content", "") for d in documents]

            response = await client.rerank(
                query=query,
                documents=texts,
                model=self.default_model,
                top_n=config.top_k,
            )

            reranked_docs = []
            for result in response.results:
                doc = dict(documents[result.index])
                doc["rerank_score"] = result.relevance_score
                doc["score"] = result.relevance_score
                reranked_docs.append(doc)

            if config.score_threshold is not None:
                before_filter = len(reranked_docs)
                reranked_docs = [
                    d
                    for d in reranked_docs
                    if d.get("rerank_score", 0) >= config.score_threshold
                ]
                logger.debug(
                    f"Filtered {before_filter - len(reranked_docs)} documents "
                    f"below threshold {config.score_threshold}"
                )

            reranked_docs = reranked_docs[: config.top_k]

            rerank_time = (time.time() - start_time) * 1000

            logger.info(
                f"Cohere reranking complete: "
                f"{len(reranked_docs)}/{len(documents)} documents, "
                f"{rerank_time:.2f}ms"
            )

            return RerankResult(
                documents=reranked_docs,
                reranker_used=self.reranker_type.value,
                rerank_time_ms=rerank_time,
                original_count=len(documents),
                filtered_count=len(reranked_docs),
            )

        except (RerankerAPIConfigError, RerankerDependencyError):
            raise
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            raise RerankingOperationError(
                reranker_type=self.reranker_type.value,
                query=query,
                document_count=len(documents),
                reason=str(e),
            ) from e
