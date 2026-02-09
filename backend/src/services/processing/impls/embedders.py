from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from loguru import logger
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.domain.exceptions import (
    APIKeyMissingError,
    EmbedderInitializationError,
    ModelNotInitializedError,
    EmbeddingOperationError,
)
from backend.src.domain.schemas import (
    ChunkFull,
    EmbedderConfig,
    EmbeddingResult,
    SourceFull,
)

from backend.src.domain.enums import EmbedderType, EmbeddingMode

if TYPE_CHECKING:
    pass


# =============================================================================
# Base Embedder Interface
# =============================================================================


class BaseEmbedder(ABC):
    _registry: ClassVar[dict[str, type[BaseEmbedder]]] = {}

    def __init__(self, session: AsyncSession | None = None):
        self.session = session
        self._lock = threading.RLock()

    @classmethod
    def register(cls, embedder_type: EmbedderType):
        def decorator(embedder_cls: type[BaseEmbedder]):
            cls._registry[embedder_type.value] = embedder_cls
            return embedder_cls

        return decorator

    @classmethod
    def get_embedder(
        cls,
        embedder_type: EmbedderType,
        session: AsyncSession | None = None,
    ) -> BaseEmbedder | None:
        embedder_cls = cls._registry.get(embedder_type.value)

        if embedder_cls:
            return embedder_cls(session)

        logger.warning(f"No embedder found for type: {embedder_type.value}")
        return None

    @classmethod
    def get_available_embedders(cls) -> list[str]:
        return list(cls._registry.keys())

    # Class attributes (defined in subclasses)
    embedder_type: ClassVar[EmbedderType]
    display_name: ClassVar[str]
    default_dense_model: ClassVar[str]
    vector_dimension: ClassVar[int]

    @abstractmethod
    async def embed_texts(
        self, texts: list[str], config: EmbedderConfig
    ) -> EmbeddingResult:
        pass

    @abstractmethod
    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        pass

    async def embed_chunks(
        self, chunks: list[ChunkFull], config: EmbedderConfig
    ) -> list[ChunkFull]:
        texts = [chunk.text or "" for chunk in chunks]
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            logger.warning("No valid texts to embed in chunks")
            return chunks

        result = await self.embed_texts(valid_texts, config=config)

        if result.errors:
            logger.error(f"Embedding errors: {result.errors}")
            return chunks

        if result.dense_vectors:
            for idx, vec_idx in enumerate(valid_indices):
                if idx < len(result.dense_vectors):
                    chunks[vec_idx].dense_vector = result.dense_vectors[idx]

        return chunks

    async def embed_source(
        self, source: SourceFull, config: EmbedderConfig
    ) -> SourceFull:
        if not source.chunks:
            return source

        source.chunks = await self.embed_chunks(chunks=source.chunks, config=config)

        return source

    def _create_error_result(self, error: str) -> EmbeddingResult:
        """Create an error result for consistent error handling."""
        return EmbeddingResult(
            dense_vectors=[],
            embedder_used=self.embedder_type.value,
            errors=[error],
        )


# =============================================================================
# Local Embedder Base Class
# =============================================================================


class LocalEmbedder(BaseEmbedder):
    @abstractmethod
    async def _initialize_models(self, config: EmbedderConfig) -> None:
        pass


# =============================================================================
# API Embedder Base Class
# =============================================================================


class APIEmbedder(BaseEmbedder):
    def _validate_api_key(self, config: EmbedderConfig) -> str:
        if not config.api_key:
            raise APIKeyMissingError(embedder_type=self.embedder_type.value)

        if isinstance(config.api_key, SecretStr):
            return config.api_key.get_secret_value()

        return str(config.api_key)


# =============================================================================
# FastEmbed Implementation
# =============================================================================


@BaseEmbedder.register(EmbedderType.FASTEMBED)
class FastEmbedEmbedder(LocalEmbedder):
    embedder_type: ClassVar[EmbedderType] = EmbedderType.FASTEMBED
    display_name: ClassVar[str] = "FastEmbed"
    default_dense_model: ClassVar[str] = "BAAI/bge-small-en-v1.5"
    default_sparse_model: ClassVar[str] = "prithivida/Splade_PP_en_v1"
    default_late_model: ClassVar[str] = "colbert-ir/colbertv2.0"
    vector_dimension: ClassVar[int] = 384

    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self._dense_embedder = None
        self._sparse_embedder = None
        self._late_embedder = None
        self._models_initialized = False

    async def _initialize_models(self, config: EmbedderConfig) -> None:
        """Lazy initialize FastEmbed models based on embedding mode."""
        if self._models_initialized:
            return

        with self._lock:
            if self._models_initialized:
                return

            try:
                from fastembed import TextEmbedding, SparseTextEmbedding

                dense_model = config.dense_model or self.default_dense_model
                self._dense_embedder = TextEmbedding(model_name=dense_model)
                logger.info(f"Initialized FastEmbed dense model: {dense_model}")

                if config.mode in (EmbeddingMode.HYBRID, EmbeddingMode.ADVANCED):
                    sparse_model = config.sparse_model or self.default_sparse_model
                    self._sparse_embedder = SparseTextEmbedding(model_name=sparse_model)
                    logger.info(f"Initialized FastEmbed sparse model: {sparse_model}")

                if config.mode == EmbeddingMode.ADVANCED:
                    from fastembed import LateInteractionTextEmbedding

                    late_model = config.late_model or self.default_late_model
                    self._late_embedder = LateInteractionTextEmbedding(
                        model_name=late_model
                    )
                    logger.info(f"Initialized FastEmbed late model: {late_model}")

                self._models_initialized = True

            except ImportError as e:
                raise EmbedderInitializationError(
                    embedder_type=self.embedder_type.value,
                    reason=f"FastEmbed not installed: {e}",
                    remediation="Install fastembed: pip install fastembed",
                )
            except Exception as e:
                raise EmbedderInitializationError(
                    embedder_type=self.embedder_type.value,
                    reason=str(e),
                    remediation="Check model name and availability",
                )

    async def embed_texts(
        self, texts: list[str], config: EmbedderConfig
    ) -> EmbeddingResult:
        start_time = time.time()

        try:
            await self._initialize_models(config)

            if self._dense_embedder is None:
                raise ModelNotInitializedError(embedder_type=self.embedder_type.value)

            dense_vectors: list[list[float]] = []

            logger.info(f"Embedding {len(texts)} texts with FastEmbed")

            for i in range(0, len(texts), config.batch_size):
                batch = texts[i : i + config.batch_size]

                batch_dense = list(self._dense_embedder.embed(batch))

                dense_vectors.extend([v.tolist() for v in batch_dense])

            embed_time = (time.time() - start_time) * 1000

            logger.info(f"FastEmbed embedding completed in {embed_time:.2f} ms")

            return EmbeddingResult(
                dense_vectors=dense_vectors,
                embedder_used=self.embedder_type.value,
                embed_time_ms=embed_time,
            )

        except (EmbedderInitializationError, ModelNotInitializedError):
            raise
        except Exception as e:
            logger.error(f"FastEmbed embedding failed: {e}")
            return self._create_error_result(str(e))

    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        try:
            await self._initialize_models(config)

            if self._dense_embedder is None:
                raise ModelNotInitializedError(embedder_type=self.embedder_type.value)

            vectors = list(self._dense_embedder.embed([query]))
            return vectors[0].tolist()

        except (EmbedderInitializationError, ModelNotInitializedError):
            raise
        except Exception as e:
            logger.error(f"FastEmbed query embedding failed: {e}")
            raise EmbeddingOperationError(
                operation="embed_query",
                reason=str(e),
            ) from e


# =============================================================================
# Sentence Transformers Implementation
# =============================================================================


@BaseEmbedder.register(EmbedderType.SENTENCE_TRANSFORMERS)
class SentenceTransformersEmbedder(LocalEmbedder):
    embedder_type: ClassVar[EmbedderType] = EmbedderType.SENTENCE_TRANSFORMERS
    display_name: ClassVar[str] = "Sentence Transformers"
    default_dense_model: ClassVar[str] = "all-MiniLM-L6-v2"
    vector_dimension: ClassVar[int] = 384

    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self._model = None

    async def _initialize_models(self, config: EmbedderConfig) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer

                model_name = config.dense_model or self.default_dense_model
                self._model = SentenceTransformer(model_name)
                logger.info(f"Initialized Sentence Transformers model: {model_name}")

            except ImportError as e:
                raise EmbedderInitializationError(
                    embedder_type=self.embedder_type.value,
                    reason=f"Sentence Transformers not installed: {e}",
                    remediation="Install: pip install sentence-transformers",
                )
            except Exception as e:
                raise EmbedderInitializationError(
                    embedder_type=self.embedder_type.value,
                    reason=str(e),
                    remediation="Check model name and availability",
                )

    async def embed_texts(
        self, texts: list[str], config: EmbedderConfig
    ) -> EmbeddingResult:
        start_time = time.time()

        try:
            await self._initialize_models(config)

            if self._model is None:
                raise ModelNotInitializedError(embedder_type=self.embedder_type.value)

            embeddings = await asyncio.to_thread(
                lambda: self._model.encode(
                    texts,
                    batch_size=config.batch_size,
                    normalize_embeddings=config.normalize,
                    show_progress_bar=False,
                )
            )

            dense_vectors = [v.tolist() for v in embeddings]
            embed_time = (time.time() - start_time) * 1000

            return EmbeddingResult(
                dense_vectors=dense_vectors,
                embedder_used=self.embedder_type.value,
                embed_time_ms=embed_time,
            )

        except (EmbedderInitializationError, ModelNotInitializedError):
            raise
        except Exception as e:
            logger.error(f"Sentence Transformers embedding failed: {e}")
            return self._create_error_result(str(e))

    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        result = await self.embed_texts(texts=[query], config=config)

        if result.dense_vectors:
            return result.dense_vectors[0]

        raise EmbeddingOperationError(
            operation="embed_query",
            reason="No dense vectors returned",
        )


# =============================================================================
# OpenAI Implementation
# =============================================================================


@BaseEmbedder.register(EmbedderType.OPENAI)
class OpenAIEmbedder(APIEmbedder):
    embedder_type: ClassVar[EmbedderType] = EmbedderType.OPENAI
    display_name: ClassVar[str] = "OpenAI"
    default_dense_model: ClassVar[str] = "text-embedding-3-small"
    vector_dimension: ClassVar[int] = 1536

    async def embed_texts(
        self, texts: list[str], config: EmbedderConfig
    ) -> EmbeddingResult:
        """Embed texts using OpenAI API."""
        start_time = time.time()

        try:
            import openai

            api_key = self._validate_api_key(config)

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.api_base,
            )

            model = config.dense_model or self.default_dense_model
            dense_vectors: list[list[float]] = []

            for i in range(0, len(texts), config.batch_size):
                batch = texts[i : i + config.batch_size]

                response = await client.embeddings.create(
                    model=model,
                    input=batch,
                )

                for embedding in response.data:
                    dense_vectors.append(embedding.embedding)

            embed_time = (time.time() - start_time) * 1000

            return EmbeddingResult(
                dense_vectors=dense_vectors,
                embedder_used=self.embedder_type.value,
                embed_time_ms=embed_time,
            )

        except ImportError as e:
            return self._create_error_result(f"OpenAI package not installed: {e}")
        except APIKeyMissingError:
            raise
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return self._create_error_result(str(e))

    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        result = await self.embed_texts([query], config=config)

        if result.dense_vectors:
            return result.dense_vectors[0]

        error_msg = ", ".join(result.errors) if result.errors else "Unknown error"
        raise EmbeddingOperationError(
            operation="embed_query",
            reason=error_msg,
        )


# =============================================================================
# Cohere Implementation
# =============================================================================


@BaseEmbedder.register(EmbedderType.COHERE)
class CohereEmbedder(APIEmbedder):
    """Embedder using Cohere API.
    Cohere supports different input types for documents vs queries.
    """

    embedder_type: ClassVar[EmbedderType] = EmbedderType.COHERE
    display_name: ClassVar[str] = "Cohere"
    default_dense_model: ClassVar[str] = "embed-english-v3.0"
    vector_dimension: ClassVar[int] = 1024

    async def embed_texts(
        self, texts: list[str], config: EmbedderConfig
    ) -> EmbeddingResult:
        """Embed texts using Cohere API with 'search_document' input type."""
        start_time = time.time()

        try:
            import cohere

            api_key = self._validate_api_key(config)

            client = cohere.AsyncClient(api_key=api_key)
            model = config.dense_model or self.default_dense_model

            response = await client.embed(
                texts=texts,
                model=model,
                input_type="search_document",
                truncate="END" if config.truncate else "NONE",
            )

            dense_vectors = [list(v) for v in response.embeddings]

            embed_time = (time.time() - start_time) * 1000

            return EmbeddingResult(
                dense_vectors=dense_vectors,
                embedder_used=self.embedder_type.value,
                embed_time_ms=embed_time,
            )

        except ImportError as e:
            return self._create_error_result(f"Cohere package not installed: {e}")
        except APIKeyMissingError:
            raise
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            return self._create_error_result(str(e))

    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        """Embed a single query using Cohere API with 'search_query' input type."""
        try:
            import cohere

            api_key = self._validate_api_key(config)

            client = cohere.AsyncClient(api_key=api_key)

            model = config.dense_model or self.default_dense_model

            response = await client.embed(
                texts=[query],
                model=model,
                input_type="search_query",
            )

            return list(response.embeddings[0])

        except ImportError as e:
            raise EmbeddingOperationError(
                operation="embed_query",
                reason=f"Cohere package not installed: {e}",
            )
        except APIKeyMissingError:
            raise
        except Exception as e:
            logger.error(f"Cohere query embedding failed: {e}")
            raise EmbeddingOperationError(
                operation="embed_query",
                reason=str(e),
            ) from e
