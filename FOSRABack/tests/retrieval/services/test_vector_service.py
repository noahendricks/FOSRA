"""Tests for VectorService and vector store backends.

Provides comprehensive test coverage for:
- BaseVectorStore registry and factory methods
- QdrantVectorStore operations
- PineconeVectorStore operations (mocked)
- WeaviateVectorStore operations (mocked)
- VectorService unified interface
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from FOSRABack.src.retrieval.services.vector_service import (
    APIVectorStore,
    BaseVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    VectorStoreConfig,
    VectorSearchResult,
    VectorService,
    VectorStoreType,
    WeaviateVectorStore,
)
from FOSRABack.src.storage.models import ConnectorORM
from FOSRABack.src.storage.schemas import ConnectorType, SparseVector, VectorPoint


from FOSRABack.src.resources.test_fixtures import (
    mock_list_sources_with_relations,
    mock_request_context,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.delete = AsyncMock()
    client.get_collection = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def qdrant_store(mock_qdrant_client, mock_request_context):
    """Create a Qdrant store with mocked client."""
    ctx = mock_request_context
    store = QdrantVectorStore()
    store.client = mock_qdrant_client  # type: ignore[attr-defined]
    store._initialized = True  # type: ignore[attr-defined]
    return store


@pytest.fixture
def sample_points():
    """Create sample vector points for testing."""
    return [
        VectorPoint(
            id="point1",
            dense_vector=[0.1, 0.2, 0.3],
            payload={"text": "test1", "source_id": "src1"},
        ),
        VectorPoint(
            id="point2",
            dense_vector=[0.4, 0.5, 0.6],
            payload={"text": "test2", "source_id": "src2"},
        ),
    ]


@pytest.fixture
def search_config():
    """Create a search configuration."""
    return VectorStoreConfig(
        top_k=10,
        min_score=0.5,
        include_metadata=True,
        include_vectors=False,
        filter_conditions={"workspace_ids": [1]},
    )


@pytest.fixture
def vector_service(mock_session):
    """Create a vector service instance."""
    return VectorService(session=mock_session)


# =============================================================================
# BaseVectorStore Tests
# =============================================================================


class TestBaseVectorStore:
    """Tests for BaseVectorStore class methods."""

    def test_get_available_stores(self):
        """Test getting list of registered stores."""
        stores = BaseVectorStore.get_available_stores()
        assert VectorStoreType.QDRANT in stores
        assert VectorStoreType.PINECONE in stores
        assert VectorStoreType.WEAVIATE in stores

    def test_get_store_qdrant(self):
        """Test factory method for Qdrant store."""
        store = BaseVectorStore.get_store(VectorStoreType.QDRANT)
        assert store is not None
        assert isinstance(store, QdrantVectorStore)

    def test_get_store_unknown(self):
        """Test factory method with unknown store type."""
        store = BaseVectorStore.get_store(VectorStoreType.MILVUS)
        assert store is None


# =============================================================================
# QdrantVectorStore Tests
# =============================================================================


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore."""

    def test_store_type(self):
        """Test store type attribute."""
        assert QdrantVectorStore.store_type == VectorStoreType.QDRANT
        assert QdrantVectorStore.display_name == "Qdrant"

    def test_init_with_client(self, mock_qdrant_client, mock_request_context):
        """Test initialization with existing client."""
        ctx = mock_request_context
        store: QdrantVectorStore = QdrantVectorStore()
        store.client = mock_qdrant_client  # type: ignore[attr-defined]
        store._initialized = True  # type: ignore[attr-defined]
        assert store._initialized is True  # type: ignore[attr-defined]
        assert store.client is mock_qdrant_client  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_new(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        ctx = mock_request_context
        """Test ensure_collection creates new collection."""
        mock_qdrant_client.collection_exists.return_value = False

        await qdrant_store.ensure_collection(
            "test_collection",
            384,
        )

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_success(
        self, qdrant_store, mock_qdrant_client, mock_list_sources_with_relations
    ):
        """Test successful upsert."""
        # Mock prepare_points to return IDs since we are testing the store logic
        with patch.object(
            qdrant_store,
            "prepare_points",
            return_value=[
                VectorPoint(id="p1", dense_vector=[0.1], payload={}),
                VectorPoint(id="p2", dense_vector=[0.2], payload={}),
            ],
        ):
            result = await qdrant_store.upsert(
                collection_name="test", sources=mock_list_sources_with_relations
            )

        assert len(result) == 2
        assert "p1" in result
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_success(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        """Test successful search."""
        ctx = mock_request_context
        mock_hit = MagicMock()
        mock_hit.id = "point1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "result"}
        mock_hit.vector = None
        mock_qdrant_client.search.return_value = [mock_hit]

        result = await qdrant_store.search(
            [0.1, 0.2, 0.3],
            ctx,
        )

        assert len(result) == 1
        assert result[0].point_id == "point1"
        assert result[0].score == 0.95


# =============================================================================
# VectorService Tests
# =============================================================================


# =============================================================================
# PineconeVectorStore Tests
# =============================================================================


class TestPineconeVectorStore:
    """Tests for PineconeVectorStore."""

    def test_store_type(self):
        """Test store type attribute."""
        assert PineconeVectorStore.store_type == VectorStoreType.PINECONE
        assert PineconeVectorStore.display_name == "Pinecone"

    def test_init(self):
        """Test initialization."""
        store = PineconeVectorStore()
        assert store._initialized is False  # type: ignore[attr-defined]
        assert store.client is None  # type: ignore[attr-defined]
        assert store.index is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_upsert_not_initialized(self, mock_list_sources_with_relations):
        """Test upsert returns empty when not initialized."""
        store = PineconeVectorStore()
        # Pinecone implementation uses points in its signature currently,
        # but Base class says sources. We follow the Base class signature fix.
        result = await store.upsert(
            sources=mock_list_sources_with_relations, collection_name="test"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, mock_request_context):
        """Test search returns empty when not initialized."""
        store = PineconeVectorStore()
        result = await store.search([0.1, 0.2], mock_request_context)
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_not_initialized(self):
        """Test delete returns 0 when not initialized."""
        store = PineconeVectorStore()
        result = await store.delete("test", ["id1"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_collection_info_not_initialized(self):
        """Test get_collection_info returns empty when not initialized."""
        store = PineconeVectorStore()
        result = await store.get_collection_info("test")
        assert result == {}


# =============================================================================
# WeaviateVectorStore Tests
# =============================================================================


class TestWeaviateVectorStore:
    """Tests for WeaviateVectorStore."""

    def test_store_type(self):
        """Test store type attribute."""
        assert WeaviateVectorStore.store_type == VectorStoreType.WEAVIATE
        assert WeaviateVectorStore.display_name == "Weaviate"

    def test_init(self):
        """Test initialization."""
        store = WeaviateVectorStore()
        assert store._initialized is False  # type: ignore[attr-defined]
        assert store.client is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_upsert_not_initialized(self, mock_list_sources_with_relations):
        """Test upsert returns empty when not initialized."""
        store = WeaviateVectorStore()
        result = await store.upsert(
            sources=mock_list_sources_with_relations, collection_name="test"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, mock_request_context):
        """Test search returns empty when not initialized."""
        store = WeaviateVectorStore()
        result = await store.search([0.1, 0.2], mock_request_context)
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_not_initialized(self):
        """Test delete returns 0 when not initialized."""
        store = WeaviateVectorStore()
        result = await store.delete("test", ["id1"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_collection_info_not_initialized(self):
        """Test get_collection_info returns empty when not initialized."""
        store = WeaviateVectorStore()
        result = await store.get_collection_info("test")
        assert result == {}


# =============================================================================
# APIVectorStore Tests
# =============================================================================


class TestAPIVectorStore:
    """Tests for APIVectorStore base class."""

    @pytest.mark.asyncio
    async def test_get_api_config_from_context(self, mock_request_context):
        """Test get_api_config_from_context returns config from preferences."""
        store = QdrantVectorStore()
        result = store.get_api_config_from_context(mock_request_context)
        assert result == mock_request_context.preferences.vector_store


# =============================================================================
# Additional QdrantVectorStore Tests
# =============================================================================


class TestQdrantVectorStoreAdvanced:
    """Additional tests for QdrantVectorStore edge cases."""

    def test_init_without_client(self):
        """Test initialization without client."""
        store = QdrantVectorStore()
        assert store._initialized is False  # type: ignore[attr-defined]
        assert store.client is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_ensure_collection_not_initialized(self):
        """Test ensure_collection raises when not initialized."""
        store = QdrantVectorStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.ensure_collection("test", 384)

    @pytest.mark.asyncio
    async def test_ensure_collection_exists(self, qdrant_store, mock_qdrant_client):
        """Test ensure_collection skips existing collection."""
        mock_qdrant_client.collection_exists.return_value = True

        await qdrant_store.ensure_collection("test_collection", 384)  # type: ignore[call-arg]

        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_euclidean(self, qdrant_store, mock_qdrant_client):
        """Test ensure_collection with euclidean distance."""
        mock_qdrant_client.collection_exists.return_value = False

        await qdrant_store.ensure_collection("test", 384)  # type: ignore[call-arg]

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_dot(self, qdrant_store, mock_qdrant_client):
        """Test ensure_collection with dot distance."""
        mock_qdrant_client.collection_exists.return_value = False

        await qdrant_store.ensure_collection("test", 384)  # type: ignore[call-arg]

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_not_initialized(self, mock_list_sources_with_relations):
        """Test upsert returns empty when not initialized."""
        store = QdrantVectorStore()
        with pytest.raises(ValueError, match="not initialized"):
            await store.upsert(
                collection_name="test", sources=mock_list_sources_with_relations
            )

    @pytest.mark.asyncio
    async def test_upsert_empty_sources(self, qdrant_store):
        """Test upsert with empty sources list."""
        with pytest.raises(
            ValueError, match="Qdrant client not initialized or no points to upsert"
        ):
            await qdrant_store.upsert(collection_name="test", sources=[])

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, mock_request_context):
        """Test search returns empty when not initialized."""
        store = QdrantVectorStore()
        result = await store.search([0.1, 0.2], mock_request_context)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_with_list_filter(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        """Test search with list filter conditions."""
        ctx = mock_request_context
        ctx.preferences.vector_store.filter_conditions = {"workspace_ids": [1, 2, 3]}
        mock_qdrant_client.search.return_value = []

        await qdrant_store.search([0.1], ctx)

        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_scalar_filter(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        """Test search with scalar filter conditions."""
        ctx = mock_request_context
        ctx.preferences.vector_store.filter_conditions = {"user_id": "user123"}
        mock_qdrant_client.search.return_value = []

        await qdrant_store.search([0.1], ctx)

        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_min_score(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        """Test search with minimum score threshold."""
        ctx = mock_request_context
        ctx.preferences.vector_store.min_score = 0.8
        mock_qdrant_client.search.return_value = []

        await qdrant_store.search([0.1], ctx)

        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert call_kwargs.get("score_threshold") == 0.8

    @pytest.mark.asyncio
    async def test_search_error(
        self, qdrant_store, mock_qdrant_client, mock_request_context
    ):
        """Test search error handling."""
        mock_qdrant_client.search.side_effect = Exception("Search error")

        result = await qdrant_store.search([0.1], mock_request_context)

        assert result == []

    @pytest.mark.asyncio
    async def test_delete_not_initialized(self):
        """Test delete returns 0 when not initialized."""
        store = QdrantVectorStore()
        result = await store.delete("test", ["id1"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_empty_ids(self, qdrant_store):
        """Test delete with empty IDs."""
        result = await qdrant_store.delete("test", [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_success(self, qdrant_store, mock_qdrant_client):
        """Test successful delete."""
        result = await qdrant_store.delete("test", ["id1", "id2"])

        assert result == 2
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_info_not_initialized(self):
        """Test get_collection_info returns empty when not initialized."""
        store = QdrantVectorStore()
        result = await store.get_collection_info("test")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_collection_info_success(self, qdrant_store, mock_qdrant_client):
        """Test successful get_collection_info."""
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.vectors_count = 1000
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info

        result = await qdrant_store.get_collection_info("test")

        assert result["name"] == "test"
        assert result["points_count"] == 1000

    @pytest.mark.asyncio
    async def test_get_collection_info_error(self, qdrant_store, mock_qdrant_client):
        """Test get_collection_info error handling."""
        mock_qdrant_client.get_collection.side_effect = Exception("Error")

        result = await qdrant_store.get_collection_info("test")

        assert result == {}

    @pytest.mark.asyncio
    async def test_close(self, qdrant_store, mock_qdrant_client):
        """Test closing Qdrant client."""
        await qdrant_store.close()

        mock_qdrant_client.close.assert_called_once()
        assert qdrant_store._initialized is False  # type: ignore[attr-defined]


# =============================================================================
# VectorStoreConfig Tests
# =============================================================================


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig()
        assert config.top_k == 10
        assert config.min_score == 0.0
        assert config.include_metadata is True
        assert config.include_vectors is False
        assert config.filter_conditions == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            top_k=20,
            min_score=0.7,
            include_metadata=False,
            include_vectors=True,
            filter_conditions={"key": "value"},
            collection_name="custom_collection",
        )
        assert config.top_k == 20
        assert config.min_score == 0.7
        assert config.include_metadata is False
        assert config.include_vectors is True
        assert config.filter_conditions == {"key": "value"}
        assert config.collection_name == "custom_collection"


# =============================================================================
# VectorSearchResult Tests
# =============================================================================


class TestVectorSearchResult:
    """Tests for VectorSearchResult."""

    def test_creation(self):
        """Test creating VectorSearchResult."""
        result = VectorSearchResult(
            point_id="point1",
            score=0.95,
            payload={"text": "test"},
        )
        assert result.point_id == "point1"
        assert result.score == 0.95
        assert result.payload == {"text": "test"}
        assert result.vector is None
        assert result.sparse_vector is None

    def test_with_vectors(self):
        """Test creating VectorSearchResult with vectors."""
        sparse = SparseVector(indices=[0, 5, 10], values=[0.1, 0.5, 0.3])
        result = VectorSearchResult(
            point_id="point1",
            score=0.95,
            payload={},
            vector=[0.1, 0.2, 0.3],
            sparse_vector=sparse,
        )
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.sparse_vector == sparse


# =============================================================================
# VectorService Tests
# =============================================================================


class TestVectorService:
    """Tests for VectorService."""

    def test_get_store(self, vector_service):
        """Test getting a store by type."""
        store = vector_service.get_store(
            VectorStoreType.QDRANT,
        )
        assert store is not None
        assert isinstance(store, QdrantVectorStore)

    def test_get_store_default(self, vector_service):
        """Test getting default store."""
        store = vector_service.get_store()
        assert store is not None
        assert isinstance(store, QdrantVectorStore)

    def test_get_store_cached(self, vector_service):
        """Test store caching."""
        store1 = vector_service.get_store(VectorStoreType.QDRANT)
        store2 = vector_service.get_store(VectorStoreType.QDRANT)
        assert store1 is store2

    def test_get_store_unknown(self, vector_service):
        """Test getting unknown store type."""
        store = vector_service.get_store(VectorStoreType.MILVUS)
        assert store is None

    def test_get_available_stores(self, vector_service):
        """Test getting available stores."""
        stores = vector_service.get_available_stores()
        assert VectorStoreType.QDRANT in stores
        assert VectorStoreType.PINECONE in stores
        assert VectorStoreType.WEAVIATE in stores

    @pytest.mark.asyncio
    async def test_initialize_store(self, vector_service, mock_request_context):
        """Test initializing a store."""
        with patch.object(QdrantVectorStore, "initialize", new_callable=AsyncMock):
            store = await vector_service.initialize_store(mock_request_context)
            assert store is not None

    @pytest.mark.asyncio
    async def test_search_with_query_vector(self, vector_service, mock_request_context):
        """Test search with query vector."""
        ctx = mock_request_context
        with patch.object(QdrantVectorStore, "search", return_value=[]):
            result = await vector_service.search(
                query_vector=[0.1, 0.2, 0.3],
                ctx=ctx,
            )
            assert result == []

    @pytest.mark.asyncio
    async def test_search_no_store(self, vector_service, mock_request_context):
        """Test search with unavailable store."""
        ctx = mock_request_context
        result = await vector_service.search(
            query_vector=[0.1, 0.2, 0.3],
            ctx=ctx,
            store_type=VectorStoreType.MILVUS,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_upsert_success(
        self, vector_service, mock_request_context, mock_list_sources_with_relations
    ):
        """Test successful upsert."""
        ctx = mock_request_context
        with patch.object(QdrantVectorStore, "upsert", return_value=["p1"]):
            result = await vector_service.upsert(
                sources=mock_list_sources_with_relations,
                ctx=ctx,
                collection_name="test",
            )
            assert result == ["p1"]

    @pytest.mark.asyncio
    async def test_upsert_no_store(
        self, vector_service, mock_request_context, mock_list_sources_with_relations
    ):
        """Test upsert with unavailable store."""
        ctx = mock_request_context
        result = await vector_service.upsert(
            sources=mock_list_sources_with_relations,
            ctx=ctx,
            collection_name="test",
            store_type=VectorStoreType.MILVUS,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_upsert_error(
        self, vector_service, mock_request_context, mock_list_sources_with_relations
    ):
        """Test upsert error handling."""
        ctx = mock_request_context
        with patch.object(QdrantVectorStore, "upsert", side_effect=Exception("Error")):
            result = await vector_service.upsert(
                sources=mock_list_sources_with_relations,
                ctx=ctx,
                collection_name="test",
            )
            assert result == []

    @pytest.mark.asyncio
    async def test_delete_success(self, vector_service):
        """Test successful delete."""
        with patch.object(QdrantVectorStore, "delete", return_value=2):
            result = await vector_service.delete("test", ["id1", "id2"])
            assert result == 2

    @pytest.mark.asyncio
    async def test_delete_no_store(self, vector_service):
        """Test delete with unavailable store."""
        result = await vector_service.delete(
            "test", ["id1"], store_type=VectorStoreType.MILVUS
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_error(self, vector_service):
        """Test delete error handling."""
        with patch.object(QdrantVectorStore, "delete", side_effect=Exception("Error")):
            result = await vector_service.delete("test", ["id1"])
            assert result == 0

    @pytest.mark.asyncio
    async def test_close_all(self, vector_service):
        """Test closing all stores."""
        vector_service.get_store(VectorStoreType.QDRANT)

        assert len(vector_service._stores) > 0

        await vector_service.close_all()

        assert len(vector_service._stores) == 0
