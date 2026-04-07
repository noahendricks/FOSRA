from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger
from qdrant_client.models import SparseVector

from backend.src.api.schemas.base import BaseModelFlex
from backend.src.domain.schemas.doc import Chunk

if TYPE_CHECKING:
    from backend.src.settings import EmbedderConfig


class EmbedderProvider(Protocol):
    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]: ...
    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None: ...


class EmbeddedQueries(BaseModelFlex):
    dense: list[float] = []
    sparse: SparseVector | None = None


# =============================================================================
# FastEmbed Provider
# =============================================================================
from fastembed import SparseTextEmbedding, TextEmbedding


class FastEmbedProvider:
    _dense_cache: dict[str, TextEmbedding] = {}
    _sparse_cache: dict[str, SparseTextEmbedding] = {}

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]

        await asyncio.to_thread(self._embed_dense, chunks, texts, config)
        if config.sparse_enabled and config.sparse_model:
            await asyncio.to_thread(self._embed_sparse, chunks, texts, config)

        return chunks

    def _embed_dense(
        self, chunks: list[Chunk], texts: list[str], config: EmbedderConfig
    ) -> None:
        model_name = config.dense_model
        if model_name not in self._dense_cache:
            self._dense_cache[model_name] = TextEmbedding(
                model_name=model_name,
                cache_dir=config.cache_dir.as_posix(),
                cuda=config.cuda_enabled,
            )
        model = self._dense_cache[model_name]

        embeddings = list(model.embed(documents=texts, batch_size=config.batch_size))
        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata.dense_embedding = embedding.tolist()

    def _embed_sparse(
        self, chunks: list[Chunk], texts: list[str], config: EmbedderConfig
    ) -> None:
        model_name = config.sparse_model
        assert model_name, "sparse_model must not be None when _embed_sparse is called"
        if model_name not in self._sparse_cache:
            self._sparse_cache[model_name] = SparseTextEmbedding(
                model_name=model_name,
                cache_dir=config.cache_dir.as_posix(),
                cuda=config.cuda_enabled,
            )
        model = self._sparse_cache[model_name]

        embeddings = list(model.embed(documents=texts, batch_size=config.batch_size))
        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata.sparse_embedding = embedding.as_object()

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model_name = config.dense_model
        if model_name not in self._dense_cache:
            self._dense_cache[model_name] = TextEmbedding(
                model_name=model_name,
                cache_dir=config.cache_dir.as_posix(),
                cuda=config.cuda_enabled,
            )
        dense_model = self._dense_cache[model_name]

        def _embed():
            dense = list(dense_model.query_embed(query))
            sparse = None
            if config.sparse_enabled and config.sparse_model:
                sparse_model_name = config.sparse_model
                if sparse_model_name not in self._sparse_cache:
                    self._sparse_cache[sparse_model_name] = SparseTextEmbedding(
                        model_name=sparse_model_name,
                        cache_dir=config.cache_dir.as_posix(),
                        cuda=config.cuda_enabled,
                    )
                sparse_model = self._sparse_cache[sparse_model_name]
                sparse_raw = list(sparse_model.query_embed(query))
                if sparse_raw:
                    d = sparse_raw[0].as_object()
                    sparse = SparseVector(
                        indices=d["indices"].tolist(), values=d["values"].tolist()
                    )
            return dense, sparse

        dense, sparse = await asyncio.to_thread(_embed)

        result = EmbeddedQueries()
        result.dense = dense[0].tolist()
        result.sparse = sparse
        return result


# =============================================================================
# FlagEmbedding Provider (BGE-M3: dense + sparse in one forward pass)
# =============================================================================
from FlagEmbedding import BGEM3FlagModel


class FlagProvider:
    _cache: dict[str, BGEM3FlagModel] = {}

    def _get_model(self, config: EmbedderConfig) -> BGEM3FlagModel:
        model_name = config.dense_model
        if model_name not in self._cache:
            devices = ["cuda"] if config.cuda_enabled else ["cpu"]
            logger.info(
                "Initializing FlagEmbedding model: {} on {}", model_name, devices
            )
            self._cache[model_name] = BGEM3FlagModel(
                model_name,
                use_fp16=False,
                cache_dir=config.cache_dir.as_posix(),
                devices=devices,
            )
        return self._cache[model_name]

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        model = self._get_model(config)

        def _encode():
            return model.encode(
                texts,
                batch_size=config.batch_size,
                max_length=config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        result = await asyncio.to_thread(_encode)

        dense_vecs = cast("list[Any]", result["dense_vecs"])
        lexical_weights = cast("list[dict[str, float]]", result["lexical_weights"])

        for chunk, dense, sparse_weights in zip(chunks, dense_vecs, lexical_weights):
            chunk.metadata.dense_embedding = cast("list[float]", dense.tolist())
            if sparse_weights:
                chunk.metadata.sparse_embedding = SparseVector(
                    indices=[int(k) for k in sparse_weights.keys()],
                    values=list(sparse_weights.values()),
                )

        return chunks

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model = self._get_model(config)

        def _encode():
            return model.encode_queries(
                [query],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        result = await asyncio.to_thread(_encode)

        dense_vecs = cast("list[Any]", result["dense_vecs"])
        lexical_weights = cast("list[dict[str, float]]", result["lexical_weights"])

        embedded = EmbeddedQueries()
        embedded.dense = cast("list[float]", dense_vecs[0].tolist())

        if lexical_weights and lexical_weights[0]:
            lw = lexical_weights[0]
            embedded.sparse = SparseVector(
                indices=[int(k) for k in lw.keys()],
                values=list(lw.values()),
            )

        return embedded


# =============================================================================
# HuggingFace Provider (sentence-transformers)
# =============================================================================
from sentence_transformers import SentenceTransformer


class HuggingFaceProvider:
    _cache: dict[str, SentenceTransformer] = {}

    def _get_model(self, config: EmbedderConfig) -> SentenceTransformer:
        model_name = config.dense_model
        if model_name not in self._cache:
            logger.info("Initializing HuggingFace model: {}", model_name)
            self._cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=config.cache_dir.as_posix(),
            )
        return self._cache[model_name]

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        model = self._get_model(config)

        def _encode():
            return model.encode(
                texts,
                batch_size=config.batch_size,
                normalize_embeddings=config.normalize,
            )

        embeddings = await asyncio.to_thread(_encode)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata.dense_embedding = embedding.tolist()

        return chunks

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model = self._get_model(config)

        def _encode():
            emb = model.encode(
                [query],
                normalize_embeddings=config.normalize,
            )
            return emb[0].tolist()

        dense = await asyncio.to_thread(_encode)

        result = EmbeddedQueries()
        result.dense = dense
        return result


# =============================================================================
# Ollama Provider
# =============================================================================
# from langchain_ollama import OllamaEmbeddings
#
#
# class OllamaProvider:
#     _cache: dict[str, OllamaEmbeddings] = {}
#
#     def _get_model(self, config: EmbedderConfig) -> OllamaEmbeddings:
#         model_name = config.dense_model
#         if model_name not in self._cache:
#             self._cache[model_name] = OllamaEmbeddings(
#                 model=model_name,
#                 base_url=config.api_base or "http://localhost:11434",
#             )
#         return self._cache[model_name]
#
#     async def embed_chunks(
#         self, chunks: list[Chunk], config: EmbedderConfig
#     ) -> list[Chunk]:
#         if not chunks:
#             return chunks
#
#         model = self._get_model(config)
#
#         def _embed():
#             return [model.embed_query(chunk.text) for chunk in chunks]
#
#         embeddings = await asyncio.to_thread(_embed)
#
#         for chunk, embedding in zip(chunks, embeddings):
#             chunk.metadata.dense_embedding = embedding
#
#         return chunks
#
#     async def embed_query(
#         self, query: str, config: EmbedderConfig
#     ) -> EmbeddedQueries | None:
#         model = self._get_model(config)
#
#         def _embed():
#             return model.embed_query(query)
#
#         dense = await asyncio.to_thread(_embed)
#
#         result = EmbeddedQueries()
#         result.dense = dense
#         return result


# =============================================================================
# NomicCode Provider (nomic-ai/nomic-embed-code: 768d, Qwen2.5-Coder base)
# Requires prompt prefixes: "Represent this query..." for queries,
# " passage: " for code passages.
# =============================================================================
class NomicCodeProvider:
    _cache: dict[str, SentenceTransformer] = {}

    QUERY_PROMPT = "Represent this query for searching relevant code: "
    PASSAGE_PROMPT = " passage: "

    def _get_model(self, config: EmbedderConfig) -> SentenceTransformer:
        model_name = config.dense_model
        if model_name not in self._cache:
            logger.info("Initializing NomicCode model: {}", model_name)
            self._cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=config.cache_dir.as_posix(),
            )
        return self._cache[model_name]

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        model = self._get_model(config)

        def _encode():
            return model.encode(
                texts,
                batch_size=config.batch_size,
                prompt_name=None,
                prompt=self.PASSAGE_PROMPT,
                normalize_embeddings=True,
            )

        embeddings = await asyncio.to_thread(_encode)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata.dense_embedding = embedding.tolist()

        return chunks

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model = self._get_model(config)

        def _encode():
            emb = model.encode(
                [query],
                prompt_name=None,
                prompt=self.QUERY_PROMPT,
                normalize_embeddings=True,
            )
            return emb[0].tolist()

        dense = await asyncio.to_thread(_encode)

        result = EmbeddedQueries()
        result.dense = dense
        return result


# =============================================================================
# Qwen3 Embedding Provider (Qwen/Qwen3-Embedding-0.6B: 0.6B, 1024d, Qwen3 base)
# Uses task-specific prompts: "query" for queries (with instruct), no prompt for docs.
# Supports MRL (Matryoshka Representation Learning) for custom dimensions 32-1024.
# =============================================================================
class Qwen3EmbeddingProvider:
    _cache: dict[str, SentenceTransformer] = {}

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        model = self._get_model(config)

        def _encode():
            return model.encode(
                texts,
                batch_size=config.batch_size,
                prompt=None,
                normalize_embeddings=True,
            )

        embeddings = await asyncio.to_thread(_encode)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata.dense_embedding = embedding.tolist()

        return chunks

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model = self._get_model(config)

        def _encode():
            emb = model.encode(
                [query],
                prompt_name="query",
                prompt=None,
                normalize_embeddings=True,
            )
            return emb[0].tolist()

        dense = await asyncio.to_thread(_encode)

        result = EmbeddedQueries()
        result.dense = dense
        return result

    def _get_model(self, config: EmbedderConfig) -> SentenceTransformer:
        model_name = config.dense_model
        if model_name not in self._cache:
            device = "cuda" if config.cuda_enabled else "cpu"
            logger.info(
                "Initializing Qwen3 Embedding model: {} on {}", model_name, device
            )
            self._cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=config.cache_dir.as_posix(),
                tokenizer_kwargs={"padding_side": "left"},
                model_kwargs={"torch_dtype": "bfloat16"},
                device=device,
            )
        return self._cache[model_name]


# =============================================================================
# JinaCode Provider (jinaai/jina-code-embeddings-0.5b: 494M, 896d, Qwen2.5-Coder base)
# Uses task-specific prompts via prompt_name: nl2code_query / nl2code_document
# Supports asymmetric embedding (different prompts for query vs passage).
# =============================================================================
class JinaCodeProvider:
    _cache: dict[str, SentenceTransformer] = {}

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        model = self._get_model(config)

        def _encode():
            return model.encode(
                texts,
                batch_size=config.batch_size,
                prompt_name="nl2code_document",
                prompt=None,
                normalize_embeddings=True,
            )

        embeddings = await asyncio.to_thread(_encode)

        for chunk, embedding in zip(chunks, cast("Any", embeddings)):
            chunk.metadata.dense_embedding = cast("list[float]", embedding.tolist())

        return chunks

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        model = self._get_model(config)

        def _encode():
            emb = model.encode(
                [query],
                prompt_name="nl2code_query",
                prompt=None,
                normalize_embeddings=True,
            )
            return emb[0].tolist()

        dense = await asyncio.to_thread(_encode)

        result = EmbeddedQueries()
        result.dense = dense
        return result

    def _get_model(self, config: EmbedderConfig) -> SentenceTransformer:
        model_name = config.dense_model
        if model_name not in self._cache:
            device = "cuda" if config.cuda_enabled else "cpu"
            logger.info("Initializing JinaCode model: {} on {}", model_name, device)
            self._cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=config.cache_dir.as_posix(),
                tokenizer_kwargs={"padding_side": "left"},
                model_kwargs={"torch_dtype": "bfloat16"},
                device=device,
            )
        return self._cache[model_name]


# =============================================================================
# Provider Registry
# =============================================================================
_PROVIDERS: dict[str, type[EmbedderProvider]] = {
    "fastembed": FastEmbedProvider,
    "flag": FlagProvider,
    "huggingface": HuggingFaceProvider,
    "nomic_code": NomicCodeProvider,
    "jina_code": JinaCodeProvider,
    "qwen3_embedding": Qwen3EmbeddingProvider,
}


def _resolve_provider(config: EmbedderConfig) -> EmbedderProvider:
    from backend.src.domain.enums import EmbedderType

    match config.embedder_type:
        case EmbedderType.FLAG:
            provider_cls = _PROVIDERS.get("flag", FlagProvider)
        case EmbedderType.HUGGINGFACE:
            provider_cls = _PROVIDERS.get("huggingface", HuggingFaceProvider)
        case EmbedderType.NOMIC_CODE:
            provider_cls = _PROVIDERS.get("nomic_code", NomicCodeProvider)
        case EmbedderType.JINA_CODE:
            provider_cls = _PROVIDERS.get("jina_code", JinaCodeProvider)
        case EmbedderType.QWEN3_EMBEDDING:
            provider_cls = _PROVIDERS.get("qwen3_embedding", Qwen3EmbeddingProvider)
        case EmbedderType.FASTEMBED:
            provider_cls = _PROVIDERS.get("fastembed", FastEmbedProvider)
        case _:
            provider_cls = _PROVIDERS.get("fastembed", FastEmbedProvider)
            logger.warning(
                f"Unknown embedder type '{config.embedder_type}', defaulting to FastEmbedProvider"
            )

    return provider_cls()


# =============================================================================
# Embedding Cache — LRU cache for embedding results
# =============================================================================


class EmbeddingCache:
    """LRU cache for embedding results keyed by content hash.

    cache key = SHA256(model_name + text) so same content with different
    models produces different embeddings.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, EmbeddedQueries] = OrderedDict()
        self._max_size = max_size

    def _make_key(
        self, text: str, model_name: str, sparse_model: str | None = None
    ) -> str:
        """generate cache key from text and model configuration."""
        key_parts = f"{model_name}:{text}"
        if sparse_model:
            key_parts = f"{key_parts}:{sparse_model}"
        return hashlib.sha256(key_parts.encode()).hexdigest()

    def get(
        self, text: str, model_name: str, sparse_model: str | None = None
    ) -> EmbeddedQueries | None:
        """retrieve cached embedding or None if not found."""
        key = self._make_key(text, model_name, sparse_model)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(
        self,
        text: str,
        model_name: str,
        result: EmbeddedQueries,
        sparse_model: str | None = None,
    ) -> None:
        """store embedding result in cache with LRU eviction."""
        key = self._make_key(text, model_name, sparse_model)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result

    def clear(self) -> None:
        """clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class EmbedderService:
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    def __init__(self, cache: EmbeddingCache | None = None):
        self._cache = cache or EmbeddingCache()

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return chunks

        chunks_to_embed: list[Chunk] = []
        for chunk in chunks:
            cached = self._cache.get(
                chunk.text,
                config.dense_model,
                config.sparse_model if config.sparse_enabled else None,
            )
            if cached:
                chunk.metadata.dense_embedding = cached.dense
                if cached.sparse:
                    chunk.metadata.sparse_embedding = cached.sparse
            else:
                chunks_to_embed.append(chunk)

        if not chunks_to_embed:
            logger.debug("All {} chunks served from cache", len(chunks))
            return chunks

        logger.info(
            "Embedding {} chunks with {} ({} served from cache)",
            len(chunks_to_embed),
            config.embedder_type,
            len(chunks) - len(chunks_to_embed),
        )

        try:
            provider = _resolve_provider(config)
            embedded = await provider.embed_chunks(chunks_to_embed, config)
            for chunk in embedded:
                result = EmbeddedQueries()
                result.dense = chunk.metadata.dense_embedding
                result.sparse = chunk.metadata.sparse_embedding
                self._cache.set(
                    chunk.text,
                    config.dense_model,
                    result,
                    config.sparse_model if config.sparse_enabled else None,
                )
            return chunks
        except Exception as e:
            logger.opt(exception=True).warning("Batch embedding failed")
            raise RuntimeError(f"Embedding Failed: {e}")

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        cached = self._cache.get(
            query,
            config.dense_model,
            config.sparse_model if config.sparse_enabled else None,
        )
        if cached:
            logger.debug("Query embedding served from cache")
            return cached

        logger.debug("Embedding query with {}", config.embedder_type)

        try:
            provider = _resolve_provider(config)
            result = await provider.embed_query(query, config)
            if result:
                self._cache.set(
                    query,
                    config.dense_model,
                    result,
                    config.sparse_model if config.sparse_enabled else None,
                )
            return result
        except Exception as e:
            logger.opt(exception=True).error("Query embedding failed")
            return None


# =============================================================================
# Code-specific embedder config factory
# =============================================================================


def bge_m3_embedder_config() -> "EmbedderConfig":
    """Return an EmbedderConfig configured for BGE-M3 via FlagProvider (1024d).

    BGE-M3 outperforms Jina Code 0.5B on semantic code search, producing
    30-100% higher cosine similarity for related function names while
    maintaining competitive exact name match performance.
    """
    from backend.src.domain.enums import EmbedderType
    from backend.src.settings import EmbedderConfig, get_settings

    settings = get_settings()
    return EmbedderConfig(
        embedder_type=EmbedderType.FLAG,
        dense_model="BAAI/bge-m3",
        dense_dimensions=1024,
        batch_size=settings.embedding.batch_size,
        normalize=True,
    )


def jina_code_embedder_config() -> "EmbedderConfig":
    """Return an EmbedderConfig configured for Jina Code Embeddings (494M, 896d)."""
    from backend.src.domain.enums import EmbedderType
    from backend.src.settings import EmbedderConfig, get_settings

    settings = get_settings()
    return EmbedderConfig(
        embedder_type=EmbedderType.JINA_CODE,
        dense_model="jinaai/jina-code-embeddings-0.5b",
        dense_dimensions=896,
        batch_size=settings.embedding.batch_size,
        normalize=True,
    )


def qwen3_embedder_config() -> "EmbedderConfig":
    """Return an EmbedderConfig configured for Qwen3 Embedding (0.6B, 1024d).

    Qwen3-Embedding-0.6B outperforms JinaCode-0.5B on MTEB benchmarks
    (64.33 vs 63.22 mean task score) with similar parameter count.
    Uses 'query' prompt for queries (with instruct template) and no prompt for documents.
    Supports Matryoshka Representation Learning for custom output dimensions.
    """
    from backend.src.domain.enums import EmbedderType
    from backend.src.settings import EmbedderConfig, get_settings

    settings = get_settings()
    return EmbedderConfig(
        embedder_type=EmbedderType.QWEN3_EMBEDDING,
        dense_model="Qwen/Qwen3-Embedding-0.6B",
        dense_dimensions=1024,
        batch_size=8,
        normalize=True,
        cuda_enabled=True,
    )
