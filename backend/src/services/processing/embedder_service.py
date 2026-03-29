from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic.v1.utils import to_camel
from qdrant_client.models import SparseVector

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


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class EmbeddedQueries(_BaseModelFlex):
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
            logger.info(f"Initializing FlagEmbedding model: {model_name}")
            self._cache[model_name] = BGEM3FlagModel(
                model_name,
                use_fp16=False,
                cache_dir=config.cache_dir.as_posix(),
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

        dense_vecs = result["dense_vecs"]
        lexical_weights = result["lexical_weights"]

        for chunk, dense, sparse_weights in zip(chunks, dense_vecs, lexical_weights):
            chunk.metadata.dense_embedding = dense.tolist()
            if sparse_weights:
                chunk.metadata.sparse_embedding = SparseVector(
                    indices=list(sparse_weights.keys()),
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

        dense_vecs = result["dense_vecs"]
        lexical_weights = result["lexical_weights"]

        embedded = EmbeddedQueries()
        embedded.dense = dense_vecs[0].tolist()

        if lexical_weights and lexical_weights[0]:
            lw = lexical_weights[0]
            embedded.sparse = SparseVector(
                indices=list(lw.keys()),
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
            logger.info(f"Initializing HuggingFace model: {model_name}")
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
# Provider Registry
# =============================================================================
_PROVIDERS: dict[str, type[EmbedderProvider]] = {
    "fastembed": FastEmbedProvider,
    "flag": FlagProvider,
    "huggingface": HuggingFaceProvider,
}


def _resolve_provider(config: EmbedderConfig) -> EmbedderProvider:
    embedder_type = config.embedder_type.value.lower()
    if embedder_type == "flag":
        provider_cls = _PROVIDERS.get("flag", FlagProvider)
    elif embedder_type == "huggingface":
        provider_cls = _PROVIDERS.get("huggingface", HuggingFaceProvider)
    elif embedder_type == "fastembed":
        provider_cls = _PROVIDERS.get("fastembed", FastEmbedProvider)
    else:
        provider_cls = _PROVIDERS.get("fastembed", FastEmbedProvider)
        logger.warning(
            f"Unknown embedder type '{embedder_type}', defaulting to FastEmbedProvider"
        )

    return provider_cls()


# =============================================================================
# EmbedderService — facade that dispatches to the right provider
# =============================================================================
class EmbedderService:
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return chunks

        logger.info(f"Embedding {len(chunks)} chunks with {config.embedder_type}")

        try:
            provider = _resolve_provider(config)
            return await provider.embed_chunks(chunks, config)
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            raise RuntimeError(f"Embedding Failed: {e}")

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        logger.debug(f"Embedding query with {config.embedder_type}")

        try:
            provider = _resolve_provider(config)
            return await provider.embed_query(query, config)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None
