from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from pinecone import PineconeConfig
from pydantic import BaseModel, Field, SecretStr

from backend.src.domain.enums import VectorStoreType
from backend.src.domain.schemas.config import (
    EmbedderConfig,
    QdrantConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import Chunk, Doc
from backend.src.services.processing.embedder_service import EmbedderService

if TYPE_CHECKING:
    pass


class VectorService:
    # config and store type match case primary implementation
    # Pydantic Model as JSONB  on USER ORM
    # configs should be pulled from user table in DB and should be either populated with defaults ,none or user custom settings
    # initialize store via langchain
    # get config from user id via request

    # initial active: qdrant, pinecone, milvus ,elasticsearch,and opensearch
    # <- remove once exception handling present - needs to fail fast
    @staticmethod
    def _get_store(
        config: VectorStoreConfig,
    ) -> VectorStore | None:

        store_type: str | None = config.preferred_store

        match store_type:
            case VectorStoreType.QDRANT:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams

                client: QdrantClient = QdrantClient(url=config.qdrant_config.api_base)

                if not client.collection_exists(
                    collection_name="user_vector_collection"
                ):
                    _ = client.create_collection(
                        collection_name="user_vector_collection",
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )

                # !note: must pass user embedder config to this point

                embedder = EmbedderService()._get_embedder(config=EmbedderConfig())

                print(embedder)

                qdrant_store: QdrantVectorStore = QdrantVectorStore(
                    client=client,
                    collection_name="user_vector_collection",
                    embedding=embedder if embedder else FastEmbedEmbeddings(),
                )

                return qdrant_store

            case VectorStoreType.PINECONE:
                from pinecone import Pinecone, ServerlessSpec

                if not config.pinecone_config.api_key:
                    raise ValueError("No Pinecone API Key Provided")

                index_name = "langchain-test-index"

                pc: Pinecone = Pinecone(api_key=config.pinecone_config.api_key)

                # index check (create if doesn't exist)
                if not pc.has_index(index_name):
                    _ = pc.create_index(
                        name=index_name,
                        dimension=1536,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )

                index = pc.Index(index_name)

                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

                from langchain_pinecone import PineconeVectorStore

                pinecone_store: VectorStore = PineconeVectorStore(
                    index=index, embedding=embeddings
                )

                return pinecone_store
            case _:
                pass
                # choose action (store, search, delete)
                # initialize qdrant store
                # pass store to action wrapper function
                # return vector id's and success or fail
            # etc,etc,etc..

    #  langchain vector stores:

    # action wrapper functions
    @staticmethod
    async def upsert(config: VectorStoreConfig, chunks: list[Chunk]) -> list[str]:
        #  Config [holds preferred and all user config for each -- if a user changes any field it updates on their pref JSONB field]
        store = VectorService._get_store(config)

        if not store:
            raise ValueError("No Vector Store to Perform Action")

        logger.info(f"Upserting {len(chunks)} chunks via {config.preferred_store}")

        try:
            ids: list[str] = await store.aadd_texts(
                texts=[c.text for c in chunks],
                metadatas=[c.metadata.model_dump() for c in chunks],
            )
            return ids
        except Exception as e:
            raise RuntimeError(f"Fatal Error Upserting via {config.preferred_store}")

    @staticmethod
    async def search(config: VectorStoreConfig, query: str):
        store: VectorStore | None = VectorService._get_store(config)

        from langchain_milvus import Milvus

        if not store:
            raise ValueError("No Vector Store to Perform Actino")

        match store:
            case QdrantVectorStore():
                # get user's current retrieval mode from config
                retrieval_mode: str = config.qdrant_config.retrieval_mode

                # search using retrieval mode
                results: list[tuple[Document, float]] = (
                    await store.asimilarity_search_with_score(
                        query=query, search_type=retrieval_mode
                    )
                )

                return results

            case PineconeVectorStore():
                pass
            case Milvus():
                pass

            # case elasticsearch

            # case opensearch

            case VectorStore():
                # all others - try except generic function - throw exception if error
                pass

    @staticmethod
    def delete(config: VectorStoreConfig):
        store = VectorService._get_store(config)

        if not store:
            raise ValueError("No Vector Store to Perform Actino")
        # delete vectors = vector_store.delete
        pass
