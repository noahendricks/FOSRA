from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.vectorstores.base import VectorStore
from loguru import logger
from qdrant_client.models import ScoredPoint

from backend.src.api.schemas.base import BaseModelFlex
from backend.src.domain.enums import RetrievalMode, VectorStoreType
from backend.src.domain.schemas.doc import Subsection
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.settings import EmbedderConfig, VectorStoreConfig

if TYPE_CHECKING:
    pass


from qdrant_client import AsyncQdrantClient, QdrantClient, models

CHUNKS_COLLECTION = "chunks"
RRF_K = 60


def _weighted_rrf_fuse(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    chunk_scores: dict[str, tuple[RetrievedChunk, float]] = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk.payload.get("chunk_id") or str(hash(chunk.text))
        score = (1.0 / (RRF_K + rank)) * dense_weight
        if key in chunk_scores:
            chunk_scores[key] = (chunk_scores[key][0], chunk_scores[key][1] + score)
        else:
            chunk_scores[key] = (chunk, score)

    for rank, chunk in enumerate(sparse_results):
        key = chunk.payload.get("chunk_id") or str(hash(chunk.text))
        score = (1.0 / (RRF_K + rank)) * sparse_weight
        if key in chunk_scores:
            chunk_scores[key] = (chunk_scores[key][0], chunk_scores[key][1] + score)
        else:
            chunk_scores[key] = (chunk, score)

    fused = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in fused[:top_k]]


class RetrievedChunk(BaseModelFlex):
    text: str
    token_count: int
    start_char: int
    score: float
    payload: dict[str, Any]


class VectorService:
    @staticmethod
    async def ensure_collection(
        client: AsyncQdrantClient, embedder_config: EmbedderConfig
    ) -> None:
        if not await client.collection_exists(CHUNKS_COLLECTION):
            _ = await client.create_collection(
                collection_name=CHUNKS_COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(
                        size=embedder_config.dense_dimensions,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            await client.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name="chunk_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name="section_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name="parent_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    @staticmethod
    async def upsert_chunks(
        client: AsyncQdrantClient,
        chunks: list[Subsection],
        embed_config: EmbedderConfig,
    ) -> list[models.PointStruct]:
        points = await VectorService.build_points(chunks, embed_config)

        _ = await client.upsert(collection_name=CHUNKS_COLLECTION, points=points)

        logger.info(f"Upserted {len(points)} leaf chunks to {CHUNKS_COLLECTION}")
        return points

    @staticmethod
    async def search_collection(
        client: AsyncQdrantClient,
        collection_name: str,
        embed_config: EmbedderConfig,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        embedded_queries = await EmbedderService().embed_query(
            query, config=embed_config
        )
        if not embedded_queries:
            raise RuntimeError("Query embedding failed")

        if embed_config.sparse_enabled:
            retrieval_mode = RetrievalMode.HYBRID
        else:
            retrieval_mode = RetrievalMode.STANDARD

        query_filter = None
        if filters:
            conditions = []
            if "doc_ids" in filters:
                conditions.append(
                    models.FieldCondition(
                        key="doc_id", match=models.MatchAny(any=filters["doc_ids"])
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        try:
            match retrieval_mode:
                case RetrievalMode.STANDARD:
                    results = await client.query_points(
                        collection_name=collection_name,
                        query=embedded_queries.dense,
                        query_filter=query_filter,
                        with_payload=True,
                        limit=limit,
                    )
                    return VectorService._to_retrieved_chunks(results.points)

                case RetrievalMode.HYBRID:
                    if not isinstance(embedded_queries.sparse, models.SparseVector):
                        raise RuntimeError("Sparse vector required for hybrid mode")
                    prefetch = [
                        models.Prefetch(
                            query=embedded_queries.dense, using="dense", limit=limit
                        ),
                        models.Prefetch(
                            query=embedded_queries.sparse, using="sparse", limit=limit
                        ),
                    ]
                    results = await client.query_points(
                        collection_name=collection_name,
                        prefetch=prefetch,
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        query_filter=query_filter,
                        with_payload=True,
                        limit=limit,
                    )
                    return VectorService._to_retrieved_chunks(results.points)

        except Exception as e:
            raise RuntimeError(f"Search failed on {collection_name}: {e}")

    @staticmethod
    async def weighted_search(
        client: AsyncQdrantClient,
        collection_name: str,
        embed_config: EmbedderConfig,
        query: str,
        filters: dict[str, Any] | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        embedded_queries = await EmbedderService().embed_query(
            query, config=embed_config
        )
        if not embedded_queries:
            raise RuntimeError("Query embedding failed")

        query_filter = None
        if filters:
            conditions = []
            if "doc_ids" in filters:
                conditions.append(
                    models.FieldCondition(
                        key="doc_id", match=models.MatchAny(any=filters["doc_ids"])
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        dense_results = await client.query_points(
            collection_name=collection_name,
            query=embedded_queries.dense,
            using="dense",
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        )

        sparse_results = await client.query_points(
            collection_name=collection_name,
            query=embedded_queries.sparse,
            using="sparse",
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        )

        return _weighted_rrf_fuse(
            VectorService._to_retrieved_chunks(dense_results.points),
            VectorService._to_retrieved_chunks(sparse_results.points),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=limit,
        )

    @staticmethod
    async def retrieve(
        client: AsyncQdrantClient,
        embed_config: EmbedderConfig,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        token_budget: int = 4096,
        merge_threshold: float = 0.5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> tuple[list[RetrievedChunk], set[str], str]:
        if embed_config.sparse_enabled:
            results = await VectorService.weighted_search(
                client,
                CHUNKS_COLLECTION,
                embed_config,
                query,
                filters,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                limit=top_k,
            )
        else:
            results = await VectorService.search_collection(
                client, CHUNKS_COLLECTION, embed_config, query, filters, top_k
            )

        file_ids = {
            r.payload.get("doc_id", "") for r in results if r.payload.get("doc_id")
        }

        merged_text = await VectorService.auto_merge(
            client, results, token_budget, merge_threshold
        )

        return results, file_ids, merged_text

    @staticmethod
    async def count_points(client: AsyncQdrantClient, collection_name: str) -> int:
        result = await client.count(collection_name=collection_name)
        return result.count

    @staticmethod
    async def delete_collection(
        client: AsyncQdrantClient, collection_name: str
    ) -> bool:
        if await client.collection_exists(collection_name):
            _ = await client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        return False

    @staticmethod
    def _get_store(
        config: VectorStoreConfig, embedder_config: EmbedderConfig
    ) -> VectorStore | QdrantClient | None:
        store_type: str | None = config.preferred_store

        match store_type:
            case VectorStoreType.QDRANT:
                qdrant_config = config.qdrant_config

                if qdrant_config.data_path:
                    client: QdrantClient = QdrantClient(path=qdrant_config.data_path)
                elif qdrant_config.url:
                    client = QdrantClient(url=qdrant_config.url)
                elif qdrant_config.api_base:
                    client = QdrantClient(url=qdrant_config.api_base)
                else:
                    client = QdrantClient(
                        host=qdrant_config.host,
                        port=qdrant_config.port,
                    )

                if not client.collection_exists(
                    collection_name=qdrant_config.collection_name
                ):
                    _ = client.create_collection(
                        collection_name=qdrant_config.collection_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=embedder_config.dense_dimensions,
                                distance=models.Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={"sparse": models.SparseVectorParams()},
                    )

                return client

            case _:
                pass

    @staticmethod
    async def upsert(
        config: VectorStoreConfig,
        embed_config: EmbedderConfig,
        chunks: list[Subsection],
    ) -> list[models.PointStruct] | None:
        store = VectorService._get_store(config, embed_config)

        logger.debug(f"store type: {type(store)}")
        if not store:
            raise ValueError("No Vector Store to Perform Action")

        logger.info(f"Upserting {len(chunks)} chunks via {config.preferred_store}")

        match store:
            case QdrantClient():
                logger.info("UPSERTING")
                try:
                    points = await VectorService.build_points(
                        chunks=chunks,
                        embed_config=embed_config,
                    )

                    if isinstance(store, QdrantClient):
                        exec_info = store.upsert(
                            collection_name=config.qdrant_config.collection_name,
                            points=points,
                        )

                        logger.bind(
                            _structured={
                                "exec_info": exec_info,
                                "points_len": len(points),
                            }
                        ).info("Qdrant upsert complete")
                        return points

                except Exception as e:
                    raise RuntimeError(
                        f"Fatal Error Upserting via {config.preferred_store} : {e}"
                    )

            case _:
                try:
                    if isinstance(store, VectorStore):
                        ids: list[str] = await store.aadd_texts(
                            texts=[c.text for c in chunks],
                            metadatas=[c.metadata.to_dict() for c in chunks],
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
            raise ValueError("No Vector Store to Perform Action")

        match store:
            case QdrantClient():
                if embed_config.sparse_enabled:
                    retrieval_mode = RetrievalMode.HYBRID
                else:
                    retrieval_mode = RetrievalMode.STANDARD

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
                            return VectorService._to_retrieved_chunks(results.points)

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
                                    query=embedded_queries.sparse,
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

                            return VectorService._to_retrieved_chunks(results.points)

                except Exception as e:
                    raise RuntimeError(
                        f"Fatal Error While Searching using {retrieval_mode}: {e}"
                    )

            case VectorStore():
                pass

    @staticmethod
    async def delete(
        client: AsyncQdrantClient,
        collection_name: str,
        doc_id: str,
    ) -> bool:
        try:
            _ = await client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=[doc_id]),
                        )
                    ]
                ),
            )
            logger.info(
                f"Deleted all chunks for doc_id: {doc_id} from {collection_name}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for doc_id {doc_id}: {e}")
            return False

    @staticmethod
    async def build_points(
        chunks: list[Subsection], embed_config: EmbedderConfig
    ) -> list[models.PointStruct]:
        points: list[models.PointStruct] = []
        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": chunk.metadata.dense_embedding,
                        "sparse": chunk.metadata.sparse_embedding,
                    },
                    payload={
                        "text": chunk.text,
                        "section_id": chunk.metadata.section_id,
                        "parent_id": chunk.metadata.parent_id,
                        "doc_id": chunk.metadata.doc_id,
                        "doc_title": chunk.metadata.doc_title,
                        "token_count": chunk.metadata.token_count,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "section_heading": chunk.metadata.section_heading,
                        "heading_level": chunk.metadata.heading_level,
                        "heading_path": chunk.metadata.heading_path,
                    },
                )
            )
        return points

    @staticmethod
    @staticmethod
    async def auto_merge(
        client: AsyncQdrantClient,
        results: list[RetrievedChunk],
        token_budget: int,
        merge_threshold: float = 0.5,
    ) -> str:
        if not results:
            return ""

        parent_groups: dict[str, list[RetrievedChunk]] = {}
        no_parent: list[RetrievedChunk] = []

        for chunk in results:
            pid = chunk.payload.get("parent_id")
            if pid:
                parent_groups.setdefault(pid, []).append(chunk)
            else:
                no_parent.append(chunk)

        node_ret: list[tuple[str, int, int]] = []
        tokens_used = 0

        for pid, siblings in parent_groups.items():
            if tokens_used >= token_budget:
                break

            cond1 = len(siblings) >= 2
            if not cond1:
                for chunk in siblings:
                    if tokens_used >= token_budget:
                        break
                    node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
                    tokens_used += chunk.token_count
                continue

            covered_chars = sum(
                c.payload["end_char"] - c.payload["start_char"]
                for c in siblings
                if c.payload.get("start_char") is not None
            )
            parent_tokens_estimate = covered_chars // 4
            parent_start = siblings[0].payload.get("start_char", 0)
            parent_chars = covered_chars / merge_threshold
            theta_star = (parent_chars / 3) * (1 + tokens_used / max(token_budget, 1))
            cond2 = covered_chars >= theta_star
            cond3 = (token_budget - tokens_used) >= parent_tokens_estimate

            if cond2 and cond3:
                records, _ = await client.scroll(
                    collection_name=CHUNKS_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="section_id",
                                match=models.MatchValue(value=pid),
                            )
                        ]
                    ),
                    with_payload=True,
                    limit=1,
                )
                parent_text: str | None = None
                for record in records:
                    if record.payload:
                        parent_text = dict(record.payload).get("text")
                        break

                if parent_text:
                    node_ret.append((parent_text, parent_tokens_estimate, parent_start))
                    tokens_used += parent_tokens_estimate
                else:
                    for chunk in siblings:
                        if tokens_used >= token_budget:
                            break
                        node_ret.append(
                            (chunk.text, chunk.token_count, chunk.start_char)
                        )
                        tokens_used += chunk.token_count
            else:
                for chunk in siblings:
                    if tokens_used >= token_budget:
                        break
                    node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
                    tokens_used += chunk.token_count

        for chunk in no_parent:
            if tokens_used >= token_budget:
                break
            node_ret.append((chunk.text, chunk.token_count, chunk.start_char))
            tokens_used += chunk.token_count

        seen: set[str] = set()
        unique: list[tuple[str, int, int]] = []
        for text, tokens, start in node_ret:
            if text not in seen:
                seen.add(text)
                unique.append((text, tokens, start))

        unique.sort(key=lambda x: x[2])
        return "\n\n".join(text for text, _, _ in unique)

    @staticmethod
    def _to_retrieved_chunks(results: list[ScoredPoint]):
        retrieved_chunks = []

        for sp in results:
            if sp.payload:
                payload = dict(sp.payload)
                payload["point_id"] = str(sp.id)
                rc = RetrievedChunk(
                    score=sp.score,
                    payload=payload,
                    text=payload.get("text", "Text Error"),
                    start_char=payload.get("start_char", 0),
                    token_count=payload.get("token_count", 0),
                )
                retrieved_chunks.append(rc)

        return retrieved_chunks
