from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from pydantic import StrictStr

from backend.src.domain.enums import RetrievalMode, VectorStoreType
from backend.src.domain.schemas.config import EmbedderConfig, VectorStoreConfig
from backend.src.domain.schemas.doc import Chunk
from backend.src.services.processing.embedder_service import EmbedderService

if TYPE_CHECKING:
    pass


from qdrant_client import QdrantClient, models


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
        config: VectorStoreConfig, embedder_config: EmbedderConfig
    ) -> VectorStore | QdrantClient | None:

        store_type: str | None = config.preferred_store

        match store_type:
            case VectorStoreType.QDRANT:

                client: QdrantClient = QdrantClient(url=config.qdrant_config.api_base)

                if not client.collection_exists(
                    collection_name=config.qdrant_config.collection_name
                ):
                    _ = client.create_collection(
                        collection_name=config.qdrant_config.collection_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=embedder_config.dense_dimensions,
                                distance=models.Distance.COSINE,
                            ),
                            "late-interaction": models.VectorParams(
                                size=embedder_config.late_dimensions,
                                distance=models.Distance.COSINE,
                                multivector_config=models.MultiVectorConfig(
                                    comparator=models.MultiVectorComparator.MAX_SIM
                                ),
                                hnsw_config=models.HnswConfigDiff(
                                    m=0,  # Disable HNSW graph creation
                                ),
                            ),
                        },
                        sparse_vectors_config={"sparse": models.SparseVectorParams()},
                    )

                return client

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
    async def upsert(
        config: VectorStoreConfig, embed_config: EmbedderConfig, chunks: list[Chunk]
    ) -> list[models.PointStruct] | None:
        #  Config [holds preferred and all user config for each -- if a user changes any field it updates on their pref JSONB field]
        store = VectorService._get_store(config, embed_config)

        logger.debug(f"store type: {type(store)}")
        if not store:
            raise ValueError("No Vector Store to Perform Action")

        logger.info(f"Upserting {len(chunks)} chunks via {config.preferred_store}")

        match store:
            case QdrantClient():
                logger.info("UPSERTING")
                # FIRST CLASS: Qdrant
                try:
                    points = []

                    # upsert
                    for c in chunks:
                        vector: dict[str, list[float] | models.SparseVector] = {
                            "dense": c.metadata.dense_embedding,
                        }
                        if c.metadata.sparse_embedding:
                            vector["sparse"] = models.SparseVector(
                                indices=c.metadata.sparse_embedding["indices"],
                                values=c.metadata.sparse_embedding["values"],
                            )
                        if c.metadata.late_embedding:
                            vector["late-interaction"] = c.metadata.late_embedding

                        point = {
                            "id": c.metadata.chunk_id,
                            "vector": vector,
                            "payload": {
                                "chunk": c.text,
                                "chunk_id": c.metadata.chunk_id,
                                "doc_id": c.metadata.doc_id,
                                "page_number": c.metadata.page_number,
                                "start_index": c.metadata.start_index,
                                "end_index": c.metadata.end_index,
                                "title": c.metadata.doc_title,
                            },
                        }
                        points.append(point)
                    if isinstance(store, QdrantClient):
                        exec_info = store.upsert(
                            collection_name=config.qdrant_config.collection_name,
                            points=points,
                        )

                        logger.info(f"exec_info: {exec_info}")

                        logger.info(f"Points len: {len(points)}")
                        return points

                except Exception as e:
                    raise RuntimeError(
                        f"Fatal Error Upserting via {config.preferred_store} : {e}"
                    )

            case _:
                try:
                    # HACK: LOW EXTENSIBILITY: only qdrant currently fully implemented
                    if isinstance(store, VectorStore):
                        ids: list[str] = await store.aadd_texts(
                            texts=[c.text for c in chunks],
                            metadatas=[c.metadata.model_dump() for c in chunks],
                        )

                        return None
                except Exception as e:
                    raise RuntimeError(
                        f"Fatal Error Upserting via {config.preferred_store}: {e}"
                    )

    @staticmethod
    async def search(
        config: VectorStoreConfig, embed_config: EmbedderConfig, query: str
    ):

        store: VectorStore | QdrantClient | None = VectorService._get_store(
            config, embedder_config=embed_config
        )

        if not store:
            raise ValueError("No Vector Store to Perform Actino")

        match store:
            case QdrantClient():
                # FIRST CLASS: Qdrant

                # TODO: ADD FILTERS

                if embed_config.sparse_enabled and embed_config.late_enabled:
                    retrieval_mode = RetrievalMode.LATE
                elif embed_config.sparse_enabled and not embed_config.late_enabled:
                    retrieval_mode = RetrievalMode.HYBRID
                else:
                    retrieval_mode = RetrievalMode.STANDARD

                # search using retrieval mode

                embedded_queries = await EmbedderService().embed_query(
                    query, config=embed_config
                )

                if not embedded_queries:
                    raise RuntimeError()

                try:
                    match retrieval_mode:
                        case RetrievalMode.STANDARD:
                            logger.debug("ENTERED RETRIEVAL STANDARD")

                            results = store.query_points(
                                collection_name=config.qdrant_config.collection_name,
                                query=embedded_queries.dense,
                                with_payload=True,
                                limit=10,
                            )
                            return results

                        case RetrievalMode.HYBRID:
                            logger.debug("ENTERED RETRIEVAL HYBRID")
                            if not isinstance(
                                embedded_queries.sparse, models.SparseVector
                            ):
                                raise RuntimeError(
                                    f"Sparse Vector must be of type 'SparseVector', not {type(embedded_queries)} "
                                )
                            prefetch = [
                                models.Prefetch(
                                    query=embedded_queries.dense,
                                    using="dense",
                                    limit=10,
                                ),
                                models.Prefetch(
                                    query=embedded_queries.sparse,  # WARN: NEEDS TO BE [Indices, Values]
                                    using="sparse",
                                    limit=10,
                                ),
                            ]

                            results = store.query_points(
                                collection_name=config.qdrant_config.collection_name,
                                prefetch=prefetch,
                                query=models.FusionQuery(fusion=models.Fusion.RRF),
                                with_payload=True,
                                limit=10,
                            )

                            return results

                        case RetrievalMode.LATE:
                            logger.debug("ENTERED RETRIEVAL LATE")
                            from qdrant_client.models import SparseVector

                            if not isinstance(embedded_queries.sparse, SparseVector):
                                raise RuntimeError(
                                    f"Sparse Vector passed to Qdrant is of type {type(embedded_queries.sparse)} rather than 'SparseVector'"
                                )

                            prefetch = [
                                models.Prefetch(
                                    query=embedded_queries.dense,
                                    using="dense",
                                    limit=10,
                                ),
                                models.Prefetch(
                                    query=embedded_queries.sparse,
                                    using="sparse",
                                    limit=10,
                                ),
                            ]

                            results = store.query_points(
                                collection_name=config.qdrant_config.collection_name,
                                prefetch=prefetch,
                                query=embedded_queries.late,
                                using="late-interaction",
                                with_payload=True,
                                limit=10,
                            )

                            return results
                except Exception as e:
                    raise RuntimeError(
                        f"Fatal Error While Searching using {retrieval_mode}: {e}"
                    )

            case PineconeVectorStore():
                pass
            # case Milvus():
            #     pass
            #
            # case elasticsearch

            # case opensearch

            case VectorStore():
                # all others - try except generic function - throw exception if error
                pass

    @staticmethod
    def delete(config: VectorStoreConfig):
        store = VectorService._get_store(config, embedder_config=EmbedderConfig())

        if not store:
            raise ValueError("No Vector Store to Perform Actino")
        # delete vectors = vector_store.delete
        pass
