from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from pydantic import BaseModel, ConfigDict, StrictStr
from pydantic.v1.utils import to_camel
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import ScoredPoint

from backend.src.domain.enums import RetrievalMode, VectorStoreType
from backend.src.domain.schemas.config import EmbedderConfig, VectorStoreConfig
from backend.src.domain.schemas.doc import Chunk
from backend.src.services.processing.embedder_service import EmbedderService

if TYPE_CHECKING:
    pass


from qdrant_client import QdrantClient, models


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class RetrievedChunk(_BaseModelFlex):
    text: str
    token_count: int
    start_char: int
    score: float
    payload: dict[str, Any]


class VectorService:

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

                    points = await VectorService().build_points(
                        chunks=chunks,
                        embed_config=embed_config,
                    )

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
                    # NOTE: LOW EXTENSIBILITY: only qdrant currently fully implemented
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
                            return VectorService()._to_retrieved_chunks(results.points)

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

                            #
                            return VectorService()._to_retrieved_chunks(results.points)

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

                            results: QueryResponse = store.query_points(
                                collection_name=config.qdrant_config.collection_name,
                                prefetch=prefetch,
                                query=embedded_queries.late,
                                using="late-interaction",
                                with_payload=True,
                                limit=10,
                            )

                            return VectorService()._to_retrieved_chunks(results.points)

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

    @staticmethod
    async def build_points(
        chunks: list[Chunk], embed_config: EmbedderConfig
    ) -> list[models.PointStruct]:
        points = []
        # NOTE: Called with embedded chunks
        from backend.src.services.processing.embedder_service import EmbedderService

        try:
            for i, chunk in enumerate(chunks):
                parent = chunk.metadata.parent  # HierarchicalChunk (L2 or L1 leaf)

                # Walk up to get all ancestor text for merge candidates
                grandparent = getattr(parent, "parent", None)  # L1 if parent is L2

                points.append(
                    models.PointStruct(
                        id=str(uuid4()),
                        vector={
                            "dense": chunk.metadata.dense_embedding,
                            "sparse": chunk.metadata.sparse_embedding,
                            "late-interaction": chunk.metadata.late_embedding,
                        },
                        payload={
                            "text": chunk.text,
                            "source_id": chunk.metadata.doc_id,
                            "token_count": chunk.metadata.token_count,
                            "start_char": chunk.metadata.start_char,
                            "end_char": chunk.metadata.end_char,
                            # Hierarchy for Auto-Merge
                            "parent_text": parent.text if parent else None,
                            "parent_token_count": parent.token_count if parent else 0,
                            "parent_start_char": parent.start_char if parent else None,
                            "parent_end_char": parent.end_char if parent else None,
                            "parent_level": parent.level if parent else None,
                            "grandparent_text": grandparent.text
                            if grandparent
                            else None,
                            "grandparent_token_count": grandparent.token_count
                            if grandparent
                            else 0,
                            "grandparent_start_char": grandparent.start_char
                            if grandparent
                            else None,
                            "grandparent_level": grandparent.level
                            if grandparent
                            else None,
                            # For grouping siblings during merge check
                            "parent_id": f"{chunk.metadata.doc_id}:{parent.start_char}:{parent.end_char}"
                            if parent
                            else None,
                            "grandparent_id": f"{chunk.metadata.doc_id}:{grandparent.start_char}:{grandparent.end_char}"
                            if grandparent
                            else None,
                        },
                    )
                )

            return points
        except Exception as e:
            raise RuntimeError("Fatal Error")

    @staticmethod
    def auto_merge(
        results: list[RetrievedChunk],
        token_budget: int,
        merge_threshold: float = 0.5,  # fraction of parent's children needed
    ) -> str:
        # NOTE: Called after search with retrieved chunks to merge any possible ancestors

        print("auto")
        if not results:
            return ""

        # group retrieved chunks by parent_id
        # {parent_id: [child chunks (L1, L2 or L3)]}
        parent_groups: dict[str, list[RetrievedChunk]] = {}
        no_parent: list[RetrievedChunk] = []

        for chunk in results:
            print(type(chunk))
            pid = chunk.payload.get("parent_id")
            # append chunks on parent id
            if pid:
                parent_groups.setdefault(pid, []).append(chunk)
            else:
                no_parent.append(chunk)

        # decide what to keep: parent text or individual chunks
        node_ret: list[tuple[str, int, int]] = []  # (text, token_count, start_char)
        tokens_used = 0

        for pid, siblings in parent_groups.items():
            if tokens_used >= token_budget:
                break

            parent_text = siblings[0].payload.get("parent_text")
            parent_tokens = siblings[0].payload.get("parent_token_count", 0)
            parent_start = siblings[0].payload.get("parent_start_char", 0)
            grandparent_id = siblings[0].payload.get("grandparent_id")

            # Cond1: at least 2 siblings retrieved
            cond1 = len(siblings) >= 2

            # Cond2: covered text >= adaptive threshold
            covered_chars = sum(
                (c.payload["end_char"] - c.payload["start_char"]) for c in siblings
            )

            parent_chars = len(parent_text) if parent_text else 1
            theta_star = (parent_chars / 3) * (1 + tokens_used / token_budget)
            cond2 = covered_chars >= theta_star

            # Cond3: parent fits in remaining budget
            cond3 = (token_budget - tokens_used) >= parent_tokens

            if cond1 and cond2 and cond3 and parent_text:
                # check if we can merge further up to grandparent
                # (only attempt if grandparent exists and budget allows)
                grandparent_text = siblings[0].payload.get("grandparent_text")
                grandparent_tokens = siblings[0].payload.get(
                    "grandparent_token_count", 0
                )

                if (
                    grandparent_text
                    and grandparent_tokens <= (token_budget - tokens_used)
                    and len(siblings) >= 3
                ):  # stricter threshold for grandparent
                    node_ret.append(
                        (
                            grandparent_text,
                            grandparent_tokens,
                            siblings[0].payload.get("grandparent_start_char", 0),
                        )
                    )

                    tokens_used += grandparent_tokens
                else:
                    node_ret.append((parent_text, parent_tokens, parent_start))
                    tokens_used += parent_tokens
            else:
                # keep individual chunks
                for chunk in siblings:
                    if tokens_used >= token_budget:
                        break
                    node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
                    tokens_used += chunk.token_count

        # add orphan chunks (no parent)
        for chunk in no_parent:
            if tokens_used >= token_budget:
                break
            node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
            tokens_used += chunk.token_count

        # deduplicate and sort by position
        seen = set()
        unique = []
        for text, tokens, start in node_ret:
            if text not in seen:
                seen.add(text)
                unique.append((text, tokens, start))

        unique.sort(key=lambda x: x[2])  # sort by start_char = document order

        return "\n\n".join(text for text, _, _ in unique)

    @staticmethod
    def _to_retrieved_chunks(results: list[ScoredPoint]):
        retrieved_chunks = []

        for sp in results:
            if sp.payload:
                rc = RetrievedChunk(
                    score=sp.score,
                    payload=sp.payload,
                    text=sp.payload.get("text", "Text Error"),
                    start_char=sp.payload.get("start_char", 0),
                    token_count=sp.payload.get("token_count", 0),
                )
                retrieved_chunks.append(rc)

        return retrieved_chunks
