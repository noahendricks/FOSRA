"""Model Registry for persistent model loading.

Models are loaded once at startup and cached for the lifetime of the process.
This eliminates cold-start latency on first request and provides a single
source of truth for model instances.

Usage:
    registry = ModelRegistry.get_instance()
    embedder = registry.get_embedder(embedder_config)
    llm = registry.get_llm(llm_config)
"""

from __future__ import annotations

import asyncio
from threading import Lock
from typing import TYPE_CHECKING, Any

from loguru import logger

from backend.src.domain.enums import EmbedderType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from backend.src.settings import EmbedderConfig, LLMConfig


class AllEmbedders:
    """Container for dense/sparse/late embedders."""

    dense: Any = None
    sparse: Any = None
    late: Any = None


class ModelRegistry:
    """Singleton registry for persistent model instances.

    Thread-safe model caching with lazy initialization.
    Models are keyed by configuration hash to support multiple configs.
    """

    _instance: "ModelRegistry | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._embedders: dict[str, AllEmbedders] = {}
        self._llms: dict[str, BaseChatModel] = {}
        self._rerankers: dict[str, Any] = {}
        self._embedder_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("ModelRegistry initialized")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _embedder_cache_key(self, config: EmbedderConfig) -> str:
        """Generate cache key from embedder config."""
        return f"{config.embedder_type.value}:{config.dense_model}:{config.cache_dir}"

    def _llm_cache_key(self, config: LLMConfig) -> str:
        """Generate cache key from LLM config."""
        return f"{config.provider}:{config.model}:{config.api_base or 'default'}"

    async def get_embedder(self, config: EmbedderConfig) -> AllEmbedders:
        """Get or create embedder instance.

        Args:
            config: Embedder configuration

        Returns:
            AllEmbedders container with dense/sparse/late models
        """
        cache_key = self._embedder_cache_key(config)

        if cache_key in self._embedders:
            logger.debug("Reusing cached embedder: {}", cache_key)
            return self._embedders[cache_key]

        async with self._embedder_lock:
            if cache_key in self._embedders:
                return self._embedders[cache_key]

            logger.info("Initializing embedder: {}", cache_key)
            embedders = await self._init_embedder(config)
            self._embedders[cache_key] = embedders
            return embedders

    async def _init_embedder(self, config: EmbedderConfig) -> AllEmbedders:
        """Initialize embedder models based on config."""
        embedders = AllEmbedders()

        match config.embedder_type:
            case EmbedderType.FASTEMBED:
                from fastembed import (
                    LateInteractionTextEmbedding,
                    SparseTextEmbedding,
                    TextEmbedding,
                )

                embedders.dense = TextEmbedding(
                    model_name=config.dense_model,
                    cache_dir=config.cache_dir.as_posix(),
                    cuda=config.cuda_enabled,
                )
                logger.info(
                    "Initialized FastEmbed dense: {}",
                    config.dense_model,
                )

                if config.sparse_enabled and config.sparse_model:
                    embedders.sparse = SparseTextEmbedding(
                        model_name=config.sparse_model,
                        cache_dir=config.cache_dir.as_posix(),
                        cuda=config.cuda_enabled,
                    )
                    logger.info(
                        "Initialized FastEmbed sparse: {}",
                        config.sparse_model,
                    )

                if config.late_enabled and config.late_model:
                    embedders.late = LateInteractionTextEmbedding(
                        model_name=config.late_model,
                        cache_dir=config.cache_dir.as_posix(),
                        cuda=config.cuda_enabled,
                    )
                    logger.info(
                        "Initialized FastEmbed late: {}",
                        config.late_model,
                    )

            case _:
                raise ValueError(f"Unsupported embedder type: {config.embedder_type}")

        return embedders

    async def get_llm(self, config: LLMConfig) -> "BaseChatModel":
        """Get or create LLM instance.

        Args:
            config: LLM configuration

        Returns:
            ChatLiteLLM instance
        """
        from backend.src.services.session.utils.llm_utils import build_llm

        cache_key = self._llm_cache_key(config)

        if cache_key in self._llms:
            logger.debug("Reusing cached LLM: {}", cache_key)
            return self._llms[cache_key]

        async with self._llm_lock:
            if cache_key in self._llms:
                return self._llms[cache_key]

            logger.info("Initializing LLM: {}", cache_key)
            llm = build_llm(config)
            self._llms[cache_key] = llm
            return llm

    async def get_reranker(self, model_name: str) -> Any:
        """Get or create reranker instance.

        Args:
            model_name: Reranker model name

        Returns:
            FlashRank Ranker instance
        """
        from flashrank import Ranker

        if model_name in self._rerankers:
            logger.debug("Reusing cached reranker: {}", model_name)
            return self._rerankers[model_name]

        logger.info("Initializing reranker: {}", model_name)
        reranker = Ranker(model_name=model_name)
        self._rerankers[model_name] = reranker
        return reranker

    def clear_embedders(self) -> None:
        """Clear embedder cache (for config changes)."""
        self._embedders.clear()
        logger.info("Cleared embedder cache")

    def clear_llms(self) -> None:
        """Clear LLM cache (for config changes)."""
        self._llms.clear()
        logger.info("Cleared LLM cache")

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "embedders": len(self._embedders),
            "llms": len(self._llms),
            "rerankers": len(self._rerankers),
        }
