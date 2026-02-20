from __future__ import annotations
from filetype import guess

from pprint import pp
import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

from loguru import logger
from pydantic import StrictFloat
from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import ScoredPoint
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID
from weaviate.connect import ConnectionParams, ProtocolParams

from backend.src.domain.exceptions import InfrastructureError, VectorStorageError
from backend.src.domain.schemas import (
    PayloadShape,
    RetrievedResult,
    SourceFull,
    VectorPoint,
    VectorSearchResult,
    VectorStoreConfig,
)

from backend.src.domain.enums import VectorStoreType

if TYPE_CHECKING:
    pass


# =============================================================================
# Base Vector Store Interface
# =============================================================================


class BaseVectorStore(ABC):
    _registry: ClassVar[dict[str, type[BaseVectorStore]]] = {}

    def __init__(self, session: AsyncSession | None = None):
        self.session = session

    @classmethod
    def register(cls, store_type: VectorStoreType):
        def decorator(store_cls: type[BaseVectorStore]):
            cls._registry[store_type] = store_cls
            return store_cls

        return decorator

    @classmethod
    def get_store(
        cls,
        store_type: VectorStoreType,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> BaseVectorStore | None:
        store_cls = cls._registry.get(store_type)
        print(f"store_type {store_type} \n")
        print(f"store_cls {store_cls} \n")
        logger.debug(f"Retrieved store class {store_cls} for type {store_type}")

        if store_cls:
            return store_cls(session=session, **kwargs)

        logger.warning(f"No store found for type {store_type}")
        return None

    @classmethod
    def get_available_stores(cls) -> list[str]:
        return list(cls._registry.keys())

    # Store identification (defined in subclasses)
    store_type: ClassVar[VectorStoreType]
    display_name: ClassVar[str]

    @abstractmethod
    async def initialize(self, config: VectorStoreConfig) -> None:
        pass

    @abstractmethod
    async def ensure_collection(
        self,
        collection_name: str,
        config: VectorStoreConfig,
    ) -> None:
        pass

    @abstractmethod
    async def upsert(
        self,
        sources: list[SourceFull],
        collection_name: str,
        config: VectorStoreConfig,
    ) -> list[str]:
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        config: VectorStoreConfig,
    ) -> list[RetrievedResult]:
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> int:
        pass

    @abstractmethod
    async def get_collection_info(
        self,
        collection_name: str,
    ) -> dict[str, Any]:
        pass

    async def close(self) -> None:
        pass

    def _create_empty_results(self) -> list[RetrievedResult]:
        return []


# =============================================================================
# Qdrant Implementation
# =============================================================================


@BaseVectorStore.register(VectorStoreType.QDRANT)
class QdrantVectorStore(BaseVectorStore):
    store_type: ClassVar[VectorStoreType] = VectorStoreType.QDRANT
    display_name: ClassVar[str] = "Qdrant"

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        session: AsyncSession | None = None,
    ):
        super().__init__(session)
        self.client = client
        self._initialized = client is not None

    async def initialize(self, config: VectorStoreConfig) -> None:
        if self.client is not None:
            logger.debug("qdrant client already initialized")
            return

        try:
            from qdrant_client import AsyncQdrantClient

            self.client = AsyncQdrantClient(
                host="localhost",
                port=6333,
                api_key=config.api_key.get_secret_value() if config.api_key else None,
            )

            logger.info(f"initialized qdrant client at {config.host}:{config.port}")
            self._initialized = True

        except ImportError:
            logger.error("qdrant-client not installed")
            raise
        except Exception as e:
            logger.error(f"failed to initialize Qdrant client: {e}")
            raise

    async def ensure_collection(
        self, collection_name: str, config: VectorStoreConfig
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Qdrant client not initialized")

        from qdrant_client import models

        try:
            if not self.client:
                logger.debug(
                    f"Client not initialized for collection: {collection_name}"
                )
                raise VectorStorageError(
                    operation="ensure_collection", reason="Client not initialized"
                )

            exists = await self.client.collection_exists(collection_name)

            if not exists:
                distance_map = {
                    "cosine": models.Distance.COSINE,
                    "euclidean": models.Distance.EUCLID,
                    "dot": models.Distance.DOT,
                }

                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=config.vector_size,
                            distance=distance_map.get(
                                config.distance_metric, models.Distance.COSINE
                            ),
                        )
                    },
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.debug(f"Collection {collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to check/create collection: {e}")
            raise

    async def _prepare_points(
        self,
        sources: list[SourceFull],
        collection_name: str,
        config: VectorStoreConfig,
    ) -> list[VectorPoint]:
        points: list[VectorPoint] = []

        await self.ensure_collection(collection_name=collection_name, config=config)

        if not self.ensure_collection:
            raise ValueError("No Collection")

        logger.debug(f"Preparing points from {len(sources)} sources")

        for source_idx, source in enumerate(sources):
            if not source.chunks:
                continue

            for chunk_idx, chunk in enumerate(source.chunks):
                try:
                    if not all(
                        [
                            chunk.text,
                            chunk.source_hash,
                            chunk.source_id,
                            chunk.dense_vector,
                        ]
                    ):
                        logger.warning(
                            f"Chunk {chunk_idx} from source {source_idx} "
                            "missing required fields, skipping"
                        )
                        continue

                    payload: PayloadShape = PayloadShape(
                        chunk_id=chunk.chunk_id,
                        name=source.name,
                        source_id=chunk.source_id,
                        source_hash=chunk.source_hash,
                        origin_type=(str(source.origin_type)),
                        origin_path=source.origin_path,
                        chunk_text=chunk.text,
                        start_index=chunk.start_index,
                        end_index=chunk.end_index,
                        token_count=chunk.token_count,
                        source_name=source.name,
                        file_type=str(source.source_type)
                        if source.source_type
                        else None,
                    )

                    point = VectorPoint(
                        id=str(ULID().to_uuid()),
                        dense_vector=chunk.dense_vector,
                        payload=payload,
                    )

                    points.append(point)

                except Exception as e:
                    logger.error(
                        f"Error preparing chunk {chunk_idx} from source {source_idx}: {e}"
                    )
                    raise

        logger.success(f"Prepared {len(points)} points from {len(sources)} sources")
        return points

    async def upsert(
        self,
        sources: list[SourceFull],
        collection_name: str,
        config: VectorStoreConfig,
    ) -> list[str]:
        if not self._initialized:
            raise RuntimeError("Qdrant client not initialized")

        if not sources:
            logger.warning("No sources to upsert")
            return []

        from qdrant_client.models import PointStruct

        logger.debug(f"Upserting sources to collection: {collection_name}")

        points = await self._prepare_points(
            sources=sources,
            collection_name=collection_name,
            config=config,
        )

        if not points:
            logger.warning("No valid points to upsert after preparation")
            return []

        q_points = [
            PointStruct(
                id=p.id,
                vector={"dense": p.dense_vector} if p.dense_vector else {},
                payload=p.payload.to_dict(),
            )
            for p in points
            if p.dense_vector
        ]

        if not q_points:
            return []

        await self.client.upsert(
            collection_name=collection_name,
            points=q_points,
        )

        logger.success(f"Upserted {len(q_points)} points to {collection_name}")
        return [p.id for p in points]

    async def search(
        self,
        query_vector: list[float],
        config: VectorStoreConfig,
        query_text: str | None = None,
    ) -> list[RetrievedResult]:
        if not self._initialized:
            return self._create_empty_results()

        from qdrant_client import models

        filter_conditions = []
        pp(config.filter_conditions.items())
        for key, value in config.filter_conditions.items():
            if isinstance(value, list):
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        try:
            logger.debug("enter try on search", filter_conditions)
            if not self.client or not isinstance(self.client, AsyncQdrantClient):
                raise InfrastructureError(
                    service_name="Qdrant Client ",
                    operation="Search",
                    reason="Client not initialized",
                    remediation="Ensure the Qdrant client is properly initialized before searching.",
                )

            response: QueryResponse = await self.client.query_points(
                collection_name=config.collection_name,
                query=query_vector,
                query_filter=None,
                using="dense",
                limit=config.top_k,
                score_threshold=config.min_score if config.min_score > 0 else None,
                with_payload=config.include_metadata,
                with_vectors=config.include_vectors,
            )

            retrieved_results = []
            for idx, result in enumerate(response.points):
                payload = result.payload

                if not payload:
                    continue

                rslt = RetrievedResult(
                    chunk_id=payload.get("chunk_id", "no chunk id"),
                    query_text=query_text if query_text else "",
                    contents=payload.get(
                        "chunk_text",
                        "no chunk text",
                    ),
                    file_type=payload.get("file_type", "no file type"),
                    result_rank=idx,
                    source_name=payload.get("name", "no source name"),
                    source_id=payload.get("source_id", "no source id"),
                    similarity_score=result.score if result.score else 0.0,
                )

                retrieved_results.append(rslt)

            print(retrieved_results)

            return retrieved_results

        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return self._create_empty_results()

    async def delete(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> int:
        if not self._initialized or point_ids is None:
            return 0

        from qdrant_client import models

        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

        logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        return len(point_ids)

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        if not self._initialized:
            return {}

        try:
            info = await self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.debug("Closed Qdrant client")


# =============================================================================
# API-Based Vector Stores
# =============================================================================


class APIVectorStore(BaseVectorStore):
    pass


@BaseVectorStore.register(VectorStoreType.PINECONE)
class PineconeVectorStore(APIVectorStore):
    store_type: ClassVar[VectorStoreType] = VectorStoreType.PINECONE
    display_name: ClassVar[str] = "Pinecone"

    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self.client = None
        self.index = None
        self._initialized = False

    async def initialize(self, config: VectorStoreConfig) -> None:
        try:
            from pinecone import Pinecone

            if not config.api_key:
                raise ValueError("Pinecone API key is required")

            self.client = Pinecone(api_key=config.api_key.get_secret_value())

            if config.collection_name:
                self.index = self.client.Index(config.collection_name)

            self._initialized = True
            logger.info("Pinecone client initialized")

        except ImportError:
            logger.error("pinecone-client not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def ensure_collection(
        self, collection_name: str, config: VectorStoreConfig
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Pinecone client not initialized")

        # Pinecone indexes are typically created via console/API
        self.index = self.client.Index(collection_name)
        logger.debug(f"Set active Pinecone index to {collection_name}")

    async def upsert(
        self,
        sources: list[SourceFull],
        collection_name: str,
        config: VectorStoreConfig,
    ) -> list[str]:
        if not self._initialized or not self.index or not sources:
            return []

        vectors = []
        for source in sources:
            for chunk in source.chunks:
                if chunk.dense_vector:
                    vectors.append(
                        {
                            "id": chunk.chunk_id,
                            "values": chunk.dense_vector,
                            "metadata": {
                                "text": chunk.text,
                                "source_id": chunk.source_id,
                            },
                        }
                    )

        if not vectors:
            return []

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.index.upsert(vectors=vectors),  # type: ignore
        )

        logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
        return [v["id"] for v in vectors]

    async def search(
        self, query_vector: list[float], config: VectorStoreConfig
    ) -> list[VectorSearchResult]:
        """Search Pinecone."""
        if not self._initialized or not self.index:
            return self._create_empty_results()  # pyright: ignore

        loop = asyncio.get_running_loop()

        filter_dict = config.filter_conditions if config.filter_conditions else None

        results = await loop.run_in_executor(
            None,
            lambda: self.index.query(
                vector=query_vector,
                top_k=config.top_k,
                include_metadata=config.include_metadata,
                include_values=config.include_vectors,
                filter=filter_dict,
            ),
        )

        return [
            VectorSearchResult(
                point_id=match["id"],
                query_text="",
                score=match["score"],
                payload=match.get("metadata", {}),
                vector=match.get("values"),
            )
            for match in results.get("matches", [])  # type: ignore
            if match["score"] >= config.min_score
        ]

    async def delete(self, collection_name: str, point_ids: list[str]) -> int:
        if not self._initialized or not self.index or not point_ids:
            return 0

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.index.delete(ids=point_ids),
        )

        logger.info(f"Deleted {len(point_ids)} points from Pinecone")
        return len(point_ids)

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        if not self._initialized or not self.index:
            return {}

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: self.index.describe_index_stats(),
        )

        return {
            "name": collection_name,
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
        }



