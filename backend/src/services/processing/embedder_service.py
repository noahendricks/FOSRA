from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastembed.common.types import NumpyArray
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic.v1.utils import to_camel
from qdrant_client.models import Document, SparseVector

from backend.src.domain.enums import EmbedderType
from backend.src.domain.schemas.doc import Chunk

if TYPE_CHECKING:
    from backend.src.settings import EmbedderConfig

from fastembed import (LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding)

from backend.src.settings import EmbedderConfig


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


#! hack: needs other provider for extensibilty
class AllEmbedders(_BaseModelFlex):
    dense: TextEmbedding | None = None
    sparse: SparseTextEmbedding | None = None
    late: LateInteractionTextEmbedding | None = None


class EmbeddedQueries(_BaseModelFlex):
    dense: list = []
    sparse: SparseVector | None = None
    late: NumpyArray | Document | None = None


class EmbedderService:

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    def _get_embedders(self, config: EmbedderConfig):
        # HACK: NO EXTENSIBILITY

        #!note: will add caching if profiling shows necessary
        embedder_type = config.embedder_type

        logger.debug(f"Initializing {embedder_type} embedder with model")

        all_embedders = AllEmbedders()

        try:
            match embedder_type:
                case EmbedderType.FASTEMBED:
                    dense_model = TextEmbedding(
                        model_name=config.dense_model,
                        cache_dir=config.cache_dir.as_posix(),
                        cuda=config.cuda_enabled,
                    )

                    all_embedders.dense = dense_model

                    if config.sparse_enabled and config.sparse_model:
                        sparse_model = SparseTextEmbedding(
                            model_name=config.sparse_model,
                            cache_dir=config.cache_dir.as_posix(),
                            cuda=config.cuda_enabled,
                        )

                        all_embedders.sparse = sparse_model

                    if config.late_enabled and config.late_model:
                        late_model = LateInteractionTextEmbedding(
                            model_name=config.late_model,
                            cache_dir=config.cache_dir.as_posix(),
                            cuda=config.cuda_enabled,
                        )

                        all_embedders.late = late_model

                    if not all_embedders:
                        raise RuntimeError(
                            f"Embedders {embedder_type} was not intialized succesfully"
                        )

                    logger.info(
                        f"Initialized FASTEMBED embedders: Dense: {all_embedders.dense is not None} | Sparse: {all_embedders.sparse is not None} Late: {all_embedders.late is not None}"
                    )

                    return all_embedders
                case _:
                    # case EmbedderType.OLLAMA:
                    #     embedder = OllamaEmbeddings(
                    #         model=config.model, base_url=config.api_base
                    #     )
                    #
                    #     logger.info(
                    #         f"Initialized OLLAMA embedder with model: {config.model}"
                    #     )
                    #
                    #     if not embedder:
                    #         raise RuntimeError(
                    #             f"Embedder {embedder_type} was not intialized succesfully"
                    #         )
                    #
                    #     return embedder
                    #
                    # case EmbedderType.HUGGINGFACE:
                    #     embedder = HuggingFaceEmbeddings(
                    #         model_name=config.model,
                    #         cache_folder=None,
                    #         encode_kwargs={"normalize_embeddings": config.normalize},
                    #     )
                    #
                    #     if not embedder:
                    #         raise RuntimeError(
                    #             f"Embedder {embedder_type} was not intialized succesfully"
                    #         )
                    #
                    #     logger.info(
                    #         f"Initialized HUGGINGFACE embedder with model: {config.model}"
                    #     )
                    #     return embedder
                    return None

        except Exception as e:
            raise e

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        # HACK: NO EXTENSIBILITY

        if not chunks:
            logger.warning("No chunks provided for embedding")

            return chunks

        logger.info(f"Starting embedding {len(chunks)} chunks")

        texts = [chunk.text for chunk in chunks]

        try:
            # !hack: set to all fastembed types, split to different providers later for extensibility
            embed_models: AllEmbedders | None = self._get_embedders(config)

            if not embed_models or not embed_models.dense:
                raise ValueError()

            # dense embed chunks in-place
            await self._dense_embed(
                chunks=chunks, config=config, dense_model=embed_models.dense
            )

            if config.sparse_enabled and embed_models.sparse:
                # sparse embed chunks in-place
                await self._sparse_embed(
                    chunks=chunks, config=config, sparse_model=embed_models.sparse
                )

            if config.late_enabled and embed_models.late:
                # late embed chunks in-place
                await self._late_embed(
                    chunks=chunks, config=config, late_model=embed_models.late
                )

            logger.info(f"Successfully embedded {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.warning(
                f"Batch embedding failed, falling back to individual embedding: {e}"
            )
            raise RuntimeError(f"Embedding Failed: {e}")

    async def embed_query(
        self, query: str, config: EmbedderConfig
    ) -> EmbeddedQueries | None:
        # HACK: NO EXTENSIBILITY
        logger.debug(f"Embedding query using {config.embedder_type}")

        embed_models = self._get_embedders(config)

        if embed_models is None or embed_models.dense is None:
            raise RuntimeError()

        dense_model = embed_models.dense

        try:
            query_embeds = EmbeddedQueries()

            # DENSE
            dense = embed_models.dense.query_embed(query)

            for i in dense:
                query_embeds.dense = i.tolist()

            # SPARSE
            if embed_models.sparse:
                sparse = embed_models.sparse.query_embed(query)

                for i in sparse:
                    d = i.as_object()
                    query_embeds.sparse = SparseVector(
                        indices=d["indices"].tolist(),
                        values=d["values"].tolist(),
                    )
            # LATE
            if embed_models.late and config.late_model:
                late_results = list(embed_models.late.query_embed(query))

                query_embeds.late = late_results[0].tolist()

            logger.info(
                f"Queries Embedded: Dense: {query_embeds.dense is not None} | Sparse: {query_embeds.sparse is not None} Late: {query_embeds.late is not None}"
            )

            return query_embeds

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")

            return None

    async def _dense_embed(
        self,
        chunks: list[Chunk],
        config: EmbedderConfig,
        dense_model: TextEmbedding,  # todo: add other providers types
    ) -> list[Chunk]:
        # HACK: NO EXTENSIBILITY
        if not chunks:
            logger.warning("No chunks provided for embedding")

            return chunks

        logger.info(f"Starting dense embedding {len(chunks)} chunks")

        texts = [chunk.text for chunk in chunks]

        try:
            # !todo: add isinstance and hasattr for different langchain providers

            dense_embeddings = dense_model.embed(documents=texts)

            for chunk, embedding in zip(chunks, dense_embeddings):
                if embedding is None:
                    logger.info(
                        f"Embedding for chunk id '{chunk.metadata.chunk_id}' returned None"
                    )
                    chunk.metadata.dense_embedding = []
                elif isinstance(embedding, list):
                    chunk.metadata.dense_embedding = embedding
                elif hasattr(embedding, "tolist"):  # numpy type
                    chunk.metadata.dense_embedding = embedding.tolist()
                else:
                    chunk.metadata.dense_embedding = embedding.tolist()

            logger.info(f"Successfully embedded {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.warning(
                f"Batch Dense Embedding Failed, falling back to individual embedding: {e}"
            )
            raise RuntimeError(f"Dense Embedding Failed: {e}")

    async def _sparse_embed(
        self,
        chunks: list[Chunk],
        config: EmbedderConfig,
        sparse_model: SparseTextEmbedding,  # todo: add other providers types
    ) -> None:
        # HACK: NO EXTENSIBILITY

        if not chunks:
            logger.warning("No chunks provided for embedding")
            return None

        logger.info(f"Starting sparse embedding of {len(chunks)}chunks")

        texts = [chunk.text for chunk in chunks]

        try:
            # !todo: add isinstance and hasattr for different langchain providers

            sparse_embeddings = sparse_model.embed(documents=texts)

            for chunk, embedding in zip(chunks, sparse_embeddings):
                if embedding is None:
                    logger.info(
                        f"Embedding for chunk id '{chunk.metadata.chunk_id}' returned None"
                    )

                else:
                    chunk.metadata.sparse_embedding = embedding.as_object()

            logger.info(f"Successfully embedded {len(chunks)} chunks")
        except Exception as e:
            logger.warning(
                f"Batch Sparse Embedding Failed, falling back to individual embedding: {e}"
            )
            raise RuntimeError(f"Sparse Embedding Failed: {e}")

    async def _late_embed(
        self,
        chunks: list[Chunk],
        config: EmbedderConfig,
        late_model: LateInteractionTextEmbedding,  # todo: add other providers types
    ) -> None:
        # HACK: NO EXTENSIBILITY

        if not chunks:
            logger.warning("No chunks provided for embedding")

        logger.info(f"Starting late embedding of {len(chunks)} chunks")

        texts = [chunk.text for chunk in chunks]

        try:
            # !todo: add isinstance and hasattr for different langchain providers

            late_embeddings = late_model.embed(documents=texts)

            for chunk, embedding in zip(chunks, late_embeddings):
                if embedding is None:
                    chunk.metadata.late_embedding = []

                elif isinstance(embedding, list):
                    chunk.metadata.late_embedding = embedding
                else:
                    chunk.metadata.late_embedding = embedding.tolist()

            logger.info(f"Successfully embedded {len(chunks)} chunks")

        except Exception as e:
            logger.warning(
                f"Batch Late Embedding Failed, falling back to individual embedding: {e}"
            )
            raise RuntimeError(f"Late Embedding Failed: {e}")


# async def embed_summary(self, summary: str, config: EmbedderConfig) -> list[float]:
#     # NOTE: for future use
#
#     logger.debug(f"Embedding summary using {config.embedder_type}")
#
#     embedder = self._get_embedder(config)
#
#     if not embedder:
#         raise RuntimeError("No Embedder Available")
#
#     async with self._semaphore:
#         try:
#             embedding = await embedder.aembed_query(text=summary)
#
#             logger.debug(f"Summary embedding completed")
#
#             return embedding
#
#         except Exception as e:
#             logger.error(f"Summary embedding failed: {e}")
#             return []
