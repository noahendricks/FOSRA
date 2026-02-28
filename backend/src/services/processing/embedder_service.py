from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, List

from langchain_core.embeddings import Embeddings
from loguru import logger

from backend.src.domain.enums import EmbedderType
from backend.src.domain.schemas.doc import Chunk

if TYPE_CHECKING:
    from backend.src.domain.schemas.config import EmbedderConfig

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from backend.src.domain.schemas.config import EmbedderConfig


class EmbedderService:

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    def _get_embedder(self, config: EmbedderConfig):
        """Get a langchain embedder based on enum."""

        #!note: will add caching if profiling shows necessary

        embedder_type = config.embedder_type

        logger.debug(f"Initializing {embedder_type} embedder with model")

        try:
            match embedder_type:
                case EmbedderType.OLLAMA:
                    embedder = OllamaEmbeddings(
                        model=config.model, base_url=config.api_base
                    )

                    logger.info(
                        f"Initialized OLLAMA embedder with model: {config.model}"
                    )

                    if not embedder:
                        raise RuntimeError(
                            f"Embedder {embedder_type} was not intialized succesfully"
                        )

                    return embedder

                case EmbedderType.HUGGINGFACE:
                    embedder = HuggingFaceEmbeddings(
                        model_name=config.model,
                        cache_folder=None,
                        encode_kwargs={"normalize_embeddings": config.normalize},
                    )

                    if not embedder:
                        raise RuntimeError(
                            f"Embedder {embedder_type} was not intialized succesfully"
                        )

                    logger.info(
                        f"Initialized HUGGINGFACE embedder with model: {config.model}"
                    )
                    return embedder

                case EmbedderType.FASTEMBED:
                    embedder = FastEmbedEmbeddings(
                        model_name=config.model, cache_dir=None
                    )
                    if not embedder:
                        raise RuntimeError(
                            f"Embedder {embedder_type} was not intialized succesfully"
                        )

                    logger.info(
                        f"Initialized FASTEMBED embedder with model: {config.model}"
                    )

                    return embedder
                case _:
                    return FastEmbedEmbeddings()
        except:
            raise

    async def embed_chunks(
        self, chunks: list[Chunk], config: EmbedderConfig
    ) -> list[Chunk]:
        """Embed text chunks and return them with embeddings."""

        if not chunks:
            logger.warning("No chunks provided for embedding")

            return chunks

        logger.info(f"Starting embedding of {len(chunks)}")

        from langchain_core.embeddings import Embeddings

        texts = [chunk.text for chunk in chunks]

        try:
            embedder: Embeddings = self._get_embedder(config)

            if isinstance(embedder, Embeddings) and hasattr(
                embedder, "aembed_documents"
            ):
                embeddings = await embedder.aembed_documents(texts)
            else:
                embeddings = await asyncio.to_thread(embedder.embed_documents, texts)

            for chunk, embedding in zip(chunks, embeddings):

                if embedding is None:
                    logger.info(
                        f"Embedding for chunk id '{chunk.metadata.chunk_id}' returned None"
                    )
                    chunk.metadata.embedding = None

                elif isinstance(embedding, list):
                    chunk.metadata.embedding = str(embedding)

                elif hasattr(embedding, "tolist"):
                    chunk.metadata.embedding = str(embedding.tolist())

                else:
                    chunk.metadata.embedding = str(embedding)

            logger.info(f"Successfully embedded {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.warning(
                f"Batch embedding failed, falling back to individual embedding: {e}"
            )
            raise RuntimeError(f"Embedding Failed: {e}")

    async def embed_query(self, query: str, config: EmbedderConfig) -> list[float]:
        """Embed a search query."""

        logger.debug(f"Embedding query using {config.embedder_type}")

        embedder = self._get_embedder(config)

        if embedder is None:
            raise RuntimeError()

        async with self._semaphore:
            try:

                def embed_sync():
                    if hasattr(embedder, "embed_query"):
                        return embedder.embed_query(query)
                    else:
                        raise ValueError(
                            f"Embedder {type(embedder)} doesn't support query embedding"
                        )

                embedding = await asyncio.to_thread(embed_sync)

                logger.debug(f"Query embedding completed")

                return embedding

            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                return []

    async def embed_summary(self, summary: str, config: EmbedderConfig) -> list[float]:
        """Embed a document summary."""
        logger.debug(f"Embedding summary using {config.embedder_type}")

        embedder = self._get_embedder(config)

        if not embedder:
            raise RuntimeError("No Embedder Available")

        async with self._semaphore:
            try:

                embedding = await embedder.aembed_query(text=summary)

                logger.debug(f"Summary embedding completed")

                return embedding

            except Exception as e:
                logger.error(f"Summary embedding failed: {e}")
                return []
