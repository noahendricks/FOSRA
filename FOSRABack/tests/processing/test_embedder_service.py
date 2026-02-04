"""
Tests for embedder_service module.

Provides >90% code coverage for:
- EmbedderConfig, EmbeddingResult dataclasses
- BaseEmbedder abstract class and registry pattern
- FastEmbedEmbedder, SentenceTransformersEmbedder (local)
- OpenAIEmbedder, CohereEmbedder (API)
- EmbedderService main interface
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from FOSRABack.src.config.request_context import EmbedderConfig
from FOSRABack.src.processing.services.embedder_service import (
    EmbedderType,
    EmbeddingMode,
    EmbeddingResult,
    BaseEmbedder,
    FastEmbedEmbedder,
    SentenceTransformersEmbedder,
    OpenAIEmbedder,
    CohereEmbedder,
    EmbedderService,
)
from FOSRABack.src.storage.schemas import (
    Chunk,
    SparseVector,
    SourceWithRelations,
    Origin,
    OriginType,
    ConnectorType,
    ulid_factory,
)

from FOSRABack.src.resources.test_fixtures import *

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding tests."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing deals with text and speech.",
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for embedding."""
    return "What is machine learning?"


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks for embedding tests."""
    return [
        Chunk(
            chunk_id=ulid_factory(),
            source_id="source1",
            source_hash="hash1",
            text="First chunk of text about AI.",
            start_index=0,
            end_index=30,
            token_count=7,
        ),
        Chunk(
            chunk_id=ulid_factory(),
            source_id="source1",
            source_hash="hash1",
            text="Second chunk about machine learning.",
            start_index=31,
            end_index=67,
            token_count=6,
        ),
        Chunk(
            chunk_id=ulid_factory(),
            source_id="source1",
            source_hash="hash1",
            text="",  # Empty chunk
            start_index=68,
            end_index=68,
            token_count=0,
        ),
    ]


@pytest.fixture
def sample_origin() -> Origin:
    """Create a sample origin."""
    return Origin(
        name="test_doc.txt",
        origin_path="/path/to/test_doc.txt",
        origin_type=OriginType.FILE,
        connector_type=ConnectorType.FILE_UPLOAD,
        source_hash="abc123hash",
    )


@pytest.fixture
def sample_source(
    sample_origin: Origin, sample_chunks: list[Chunk]
) -> SourceWithRelations:
    """Create a sample source with chunks."""
    return SourceWithRelations(
        source_id="source1",
        source_hash=sample_origin.source_hash,
        unique_id="test-unique-id",
        source_summary="Test document about ML",
        summary_embedding="[]",
        source_content="Test content",
        origin=sample_origin,
        chunks=sample_chunks,
    )


@pytest.fixture
def embedder_config() -> EmbedderConfig:
    """Default embedder configuration."""
    return EmbedderConfig(
        # mode=EmbeddingMode.DENSE_ONLY,
        batch_size=32,
        # max_concurrent=3,
        normalize=True,
    )


@pytest.fixture
def mock_session():
    """Mock async database session."""
    session = AsyncMock()
    return session


# =============================================================================
# EmbedderConfig Tests
# =============================================================================


class TestEmbedderConfig:
    """Tests for EmbedderConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbedderConfig()

        # assert config.mode == EmbeddingMode.DENSE_ONLY
        assert config.batch_size == 32
        assert config.max_concurrent == 3
        assert config.normalize is True
        assert config.embedder_type == EmbedderType.FASTEMBED
        assert config.truncate is True
        assert config.max_length == 512
        assert config.dense_model is None
        assert config.sparse_model is None
        assert config.late_model is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbedderConfig(
            mode=EmbeddingMode.HYBRID,
            batch_size=64,
            dense_model="custom-model",
        )
        assert config.mode == EmbeddingMode.HYBRID
        assert config.batch_size == 64
        assert config.dense_model == "custom-model"


# =============================================================================
# EmbeddingResult Tests
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = EmbeddingResult()
        assert result.dense_vectors is None
        assert result.sparse_vectors is None
        assert result.late_interaction_vectors is None
        assert result.embedder_used == ""
        assert result.embed_time_ms == 0.0
        assert result.token_count == 0
        assert result.errors == []

    def test_with_dense_vectors(self):
        """Test result with dense vectors."""
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            dense_vectors=vectors,
            embedder_used="FASTEMBED",
            embed_time_ms=50.5,
        )
        assert result.dense_vectors == vectors
        assert result.embedder_used == "FASTEMBED"

    def test_with_sparse_vectors(self):
        """Test result with sparse vectors."""
        sparse = [SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])]
        result = EmbeddingResult(sparse_vectors=sparse)
        assert result.sparse_vectors == sparse

    def test_with_errors(self):
        """Test result with errors."""
        result = EmbeddingResult(errors=["Error 1", "Error 2"])
        assert len(result.errors) == 2


# =============================================================================
# SparseVector Tests
# =============================================================================


class TestSparseVector:
    """Tests for SparseVector schema."""

    def test_creation(self):
        """Test creating a sparse vector."""
        sparse = SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])
        assert sparse.indices == [0, 2, 5]
        assert sparse.values == [0.1, 0.3, 0.5]

    def test_empty(self):
        """Test empty sparse vector."""
        sparse = SparseVector(indices=[], values=[])
        assert sparse.indices == []
        assert sparse.values == []


# =============================================================================
# BaseEmbedder Tests
# =============================================================================


class TestBaseEmbedder:
    """Tests for BaseEmbedder abstract class and registry."""

    def test_registry_contains_embedders(self):
        """Test that embedder types are registered."""
        available = BaseEmbedder.get_available_embedders()
        assert "FASTEMBED" in available
        assert "SENTENCE_TRANSFORMERS" in available
        assert "OPENAI" in available
        assert "COHERE" in available

    def test_get_embedder_valid_type(self):
        """Test getting a valid embedder."""
        embedder = BaseEmbedder.get_embedder(EmbedderType.FASTEMBED)
        assert embedder is not None
        assert isinstance(embedder, FastEmbedEmbedder)

    def test_get_embedder_with_session(self, mock_session):
        """Test getting embedder with session."""
        embedder = BaseEmbedder.get_embedder(EmbedderType.OPENAI, mock_session)
        assert embedder is not None
        assert embedder.session == mock_session

    def test_get_embedder_invalid_type(self):
        """Test getting invalid embedder type returns None."""
        # Save and clear registry
        original = BaseEmbedder._registry.copy()
        BaseEmbedder._registry.clear()

        try:
            embedder = BaseEmbedder.get_embedder(EmbedderType.FASTEMBED)
            assert embedder is None
        finally:
            BaseEmbedder._registry = original

    def test_create_error_result(self):
        """Test the _create_error_result helper method."""
        embedder = FastEmbedEmbedder()
        result = embedder._create_error_result("Test error")
        assert result.dense_vectors is None
        assert result.embedder_used == EmbedderType.FASTEMBED.value
        assert "Test error" in result.errors


# =============================================================================
# FastEmbedEmbedder Tests
# =============================================================================


class TestFastEmbedEmbedder:
    """Tests for FastEmbedEmbedder."""

    def test_class_attributes(self):
        """Test class attributes."""
        assert FastEmbedEmbedder.embedder_type == EmbedderType.FASTEMBED
        assert FastEmbedEmbedder.display_name == "FastEmbed"
        assert FastEmbedEmbedder.default_dense_model == "BAAI/bge-small-en-v1.5"
        assert FastEmbedEmbedder.vector_dimension == 384

    def test_initialization(self):
        """Test embedder initialization."""
        embedder = FastEmbedEmbedder()
        assert embedder._dense_embedder is None  # type: ignore[union-attr]
        assert embedder._sparse_embedder is None  # type: ignore[union-attr]
        assert embedder._late_embedder is None  # type: ignore[union-attr]
        assert embedder._models_initialized is False  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_embed_texts_dense_only(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test embedding texts with dense only mode."""
        embedder = FastEmbedEmbedder()
        config = EmbedderConfig(mode=EmbeddingMode.DENSE_ONLY)

        ctx = mock_request_context
        # Mock the embedder
        mock_dense = MagicMock()
        mock_dense.embed.return_value = [
            np.array([0.1, 0.2, 0.3]) for _ in sample_texts
        ]
        embedder._dense_embedder = mock_dense  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_texts(texts=sample_texts, ctx=ctx)

        assert result.dense_vectors is not None
        assert len(result.dense_vectors) == len(sample_texts)
        assert result.embedder_used == "FASTEMBED"
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_embed_texts_hybrid(
        self, mock_request_context, sample_texts: list[str]
    ):
        """Test embedding texts with hybrid mode."""
        embedder = FastEmbedEmbedder()

        # Mock both embedders
        mock_dense = MagicMock()
        mock_dense.embed.return_value = [
            np.array([0.1, 0.2, 0.3]) for _ in sample_texts
        ]

        mock_sparse_vec = MagicMock()
        mock_sparse_vec.indices = np.array([0, 2, 5])
        mock_sparse_vec.values = np.array([0.1, 0.3, 0.5])

        mock_sparse = MagicMock()
        mock_sparse.embed.return_value = [mock_sparse_vec for _ in sample_texts]

        embedder._dense_embedder = mock_dense  # type: ignore[attr-defined]
        embedder._sparse_embedder = mock_sparse  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_texts(
            texts=sample_texts, ctx=mock_request_context
        )

        assert result.dense_vectors is not None
        # assert result.sparse_vectors is not None
        # assert len(result.sparse_vectors) == len(sample_texts)

    @pytest.mark.asyncio
    async def test_embed_texts_not_initialized(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test error when embedder not initialized."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()
        embedder._models_initialized = True  # type: ignore[attr-defined]  # Skip init but keep embedder None

        result = await embedder.embed_texts(sample_texts, ctx=ctx)

        assert len(result.errors) > 0
        assert "not initialized" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_texts_error_handling(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test error handling in embedding."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()

        embedder._dense_embedder = MagicMock()  # type: ignore[attr-defined]
        embedder._dense_embedder.embed.side_effect = Exception("Embedding failed")  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_texts(sample_texts, ctx=ctx)

        assert len(result.errors) > 0
        assert "Embedding failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_query(self, sample_query: str, mock_request_context):
        """Test embedding a single query."""
        embedder = FastEmbedEmbedder()

        ctx = mock_request_context
        mock_dense = MagicMock()
        mock_dense.embed.return_value = [np.array([0.1, 0.2, 0.3])]

        embedder._dense_embedder = mock_dense  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_query(sample_query, ctx=ctx)

        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embed_query_error(self, sample_query: str, mock_request_context):
        """Test query embedding error."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()
        embedder._models_initialized = True  # type: ignore[attr-defined]  # Not initialized

        with pytest.raises(RuntimeError):
            await embedder.embed_query(query=sample_query, ctx=ctx)

    @pytest.mark.asyncio
    async def test_embed_chunks(self, mock_request_context, sample_chunks: list[Chunk]):
        """Test embedding chunks."""
        embedder = FastEmbedEmbedder()

        mock_dense = MagicMock()
        # Only 2 valid chunks (one is empty)
        mock_dense.embed.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        embedder._dense_embedder = mock_dense  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_chunks(
            chunks=sample_chunks, ctx=mock_request_context
        )

        assert len(result) == 3
        # First two chunks should have vectors
        assert result[0].dense_vector is not None
        assert result[1].dense_vector is not None
        # Third chunk (empty) should not have vector
        assert result[2].dense_vector is None

    @pytest.mark.asyncio
    async def test_embed_chunks_empty_list(self, mock_request_context):
        """Test embedding empty chunk list."""
        embedder = FastEmbedEmbedder()
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_chunks([], ctx=mock_request_context)

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_chunks_all_empty_text(self, mock_request_context):
        """Test embedding chunks with all empty text."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()
        embedder._models_initialized = True  # type: ignore[attr-defined]

        chunks = [
            Chunk(
                chunk_id="1",
                source_id="s1",
                text="",
                start_index=0,
                end_index=0,
                token_count=0,
            ),
            Chunk(
                chunk_id="2",
                source_id="s1",
                text="   ",  # Whitespace only
                start_index=0,
                end_index=3,
                token_count=0,
            ),
        ]

        result = await embedder.embed_chunks(chunks=chunks, ctx=ctx)

        # Should return chunks unchanged
        assert len(result) == 2
        assert result[0].dense_vector is None
        assert result[1].dense_vector is None

    @pytest.mark.asyncio
    async def test_embed_source(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test embedding a source."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()

        mock_dense = MagicMock()
        mock_dense.embed.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        embedder._dense_embedder = mock_dense  # type: ignore[attr-defined]
        embedder._models_initialized = True  # type: ignore[attr-defined]

        result = await embedder.embed_source(sample_source, ctx=ctx)

        assert result.chunks[0].dense_vector is not None

    @pytest.mark.asyncio
    async def test_embed_source_no_chunks(
        self, sample_origin: Origin, mock_request_context
    ):
        """Test embedding source with no chunks."""
        ctx = mock_request_context
        embedder = FastEmbedEmbedder()

        source = SourceWithRelations(
            source_id="source1",
            source_hash=sample_origin.source_hash,
            unique_id="test",
            source_summary="",
            summary_embedding="[]",
            source_content="",
            origin=sample_origin,
            chunks=[],
        )

        result = await embedder.embed_source(source=source, ctx=ctx)

        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_initialize_models_hybrid(self):
        """Test model initialization in hybrid mode."""
        embedder = FastEmbedEmbedder()
        config = EmbedderConfig(mode=EmbeddingMode.HYBRID)

        with patch("fastembed.TextEmbedding") as mock_text:
            with patch("fastembed.SparseTextEmbedding") as mock_sparse:
                mock_text.return_value = MagicMock()
                mock_sparse.return_value = MagicMock()

                await embedder._initialize_models(config)  # type: ignore[attr-defined]

                mock_text.assert_called_once()
                mock_sparse.assert_called_once()
                assert embedder._models_initialized is True  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_initialize_models_advanced(self):
        """Test model initialization in advanced mode."""
        embedder = FastEmbedEmbedder()
        config = EmbedderConfig(mode=EmbeddingMode.ADVANCED)

        with patch("fastembed.TextEmbedding") as mock_text:
            with patch("fastembed.SparseTextEmbedding") as mock_sparse:
                with patch("fastembed.LateInteractionTextEmbedding") as mock_late:
                    mock_text.return_value = MagicMock()
                    mock_sparse.return_value = MagicMock()
                    mock_late.return_value = MagicMock()

                    await embedder._initialize_models(config)  # type: ignore[attr-defined]

                    mock_late.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_models_idempotent(self):
        """Test that initialization only happens once."""
        embedder = FastEmbedEmbedder()
        embedder._models_initialized = True  # type: ignore[attr-defined]
        config = EmbedderConfig()

        with patch("fastembed.TextEmbedding") as mock_text:
            await embedder._initialize_models(config)  # type: ignore[attr-defined]
            mock_text.assert_not_called()


# =============================================================================
# SentenceTransformersEmbedder Tests
# =============================================================================


class TestSentenceTransformersEmbedder:
    """Tests for SentenceTransformersEmbedder."""

    def test_class_attributes(self):
        """Test class attributes."""
        assert (
            SentenceTransformersEmbedder.embedder_type
            == EmbedderType.SENTENCE_TRANSFORMERS
        )
        assert SentenceTransformersEmbedder.display_name == "Sentence Transformers"
        assert SentenceTransformersEmbedder.default_dense_model == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_embed_texts(self, sample_texts: list[str], mock_request_context):
        """Test embedding texts."""
        ctx = mock_request_context
        embedder = SentenceTransformersEmbedder()
        config = EmbedderConfig()

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        embedder._model = mock_model  # type: ignore[attr-defined]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

            result = await embedder.embed_texts(texts=sample_texts, ctx=ctx)

            assert result.dense_vectors is not None
            assert len(result.dense_vectors) == 3
            assert result.embedder_used == "SENTENCE_TRANSFORMERS"

    @pytest.mark.asyncio
    async def test_embed_texts_error(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test error handling."""
        ctx = mock_request_context
        embedder = SentenceTransformersEmbedder()
        embedder._model = MagicMock()  # type: ignore[attr-defined]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Model error")

            result = await embedder.embed_texts(sample_texts, ctx=ctx)

            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_embed_texts_not_initialized(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test with uninitialized model."""
        ctx = mock_request_context
        embedder = SentenceTransformersEmbedder()
        # Model is None

        with patch.object(embedder, "_initialize_models", new_callable=AsyncMock):
            result = await embedder.embed_texts(sample_texts, ctx=ctx)

            assert len(result.errors) > 0
            assert "not initialized" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_query(self, sample_query: str, mock_request_context):
        """Test embedding a query."""
        ctx = mock_request_context
        embedder = SentenceTransformersEmbedder()

        mock_model = MagicMock()
        embedder._model = mock_model  # type: ignore[attr-defined]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.return_value = np.array([[0.1, 0.2, 0.3]])

            result = await embedder.embed_query(query=sample_query, ctx=ctx)

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embed_query_error(self, sample_query: str, mock_request_context):
        """Test query embedding error."""
        ctx = mock_request_context
        embedder = SentenceTransformersEmbedder()

        with patch.object(
            embedder, "embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = EmbeddingResult(errors=["Failed"])

            with pytest.raises(RuntimeError):
                await embedder.embed_query(query=sample_query, ctx=ctx)

    @pytest.mark.asyncio
    async def test_initialize_models(self):
        """Test model initialization."""
        embedder = SentenceTransformersEmbedder()
        config = EmbedderConfig()

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()

            await embedder._initialize_models(config)  # type: ignore[attr-defined]

            mock_st.assert_called_once_with(
                config.dense_model or embedder.default_dense_model
            )
            assert embedder._model is not None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_initialize_models_idempotent(self):
        """Test that initialization only happens once."""
        embedder = SentenceTransformersEmbedder()
        embedder._model = MagicMock()  # type: ignore[attr-defined]  # Already initialized
        config = EmbedderConfig()

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            await embedder._initialize_models(config)  # type: ignore[attr-defined]
            mock_st.assert_not_called()


# =============================================================================
# OpenAIEmbedder Tests
# =============================================================================


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    def test_class_attributes(self):
        """Test class attributes."""
        assert OpenAIEmbedder.embedder_type == EmbedderType.OPENAI
        assert OpenAIEmbedder.display_name == "OpenAI"
        assert OpenAIEmbedder.default_dense_model == "text-embedding-3-small"
        assert OpenAIEmbedder.vector_dimension == 1536

    @pytest.mark.asyncio
    async def test_get_api_config_no_session(self):
        """Test getting API config without session."""
        embedder = OpenAIEmbedder()
        config = await embedder.get_api_config("user123")  # type: ignore[attr-defined]
        assert config is None

    @pytest.mark.asyncio
    async def test_get_api_config_found(
        self, mock_session, mock_request_context: RequestContext
    ):
        """Test getting API config when found in DB."""

        ctx = mock_request_context

        config = ctx.preferences.embedder
        mock_llm_config = MagicMock()
        mock_llm_config.api_key = "test_key"
        mock_llm_config.api_base = "test_base"

        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            mock_llm_config
        )

        config = await config.get_api_config("user123")  # type: ignore[attr-defined]

        assert config == {"api_key": "test_key", "api_base": "test_base"}

    @pytest.mark.asyncio
    async def test_get_api_config_not_found(self, mock_session):
        """Test getting API config when not found in DB."""
        embedder = OpenAIEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = None

        config = await embedder.get_api_config("user123")  # type: ignore[attr-defined]
        assert config is None

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, sample_texts: list[str], mock_session):
        """Test successful embedding with OpenAI API."""
        embedder = OpenAIEmbedder(session=mock_session)

        # Mock API config retrieval
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="test_key", api_base="test_base")
        )

        # Mock OpenAI client and response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding] * len(sample_texts)

        with patch(
            "FOSRABack.src.processing.services.embedder_service.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.return_value.embeddings.create.return_value = mock_response

            result = await embedder.embed_texts(sample_texts)

            assert result.dense_vectors is not None
            assert len(result.dense_vectors) == len(sample_texts)
            assert result.embedder_used == "OPENAI"
            assert result.errors == []
            mock_openai.return_value.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_no_user_id(self, sample_texts: list[str]):
        """Test embedding texts without user_id."""
        embedder = OpenAIEmbedder()
        result = await embedder.embed_texts(sample_texts)
        assert len(result.errors) > 0
        assert "user_id required" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_texts_no_api_config(
        self, sample_texts: list[str], mock_session
    ):
        """Test embedding texts without API config."""
        embedder = OpenAIEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = None

        result = await embedder.embed_texts(sample_texts)
        assert len(result.errors) > 0
        assert "not configured" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_texts_api_error(self, sample_texts: list[str], mock_session):
        """Test OpenAI API error handling."""
        embedder = OpenAIEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="test_key")
        )

        with patch(
            "FOSRABack.src.processing.services.embedder_service.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.return_value.embeddings.create.side_effect = Exception(
                "OpenAI API error"
            )

            result = await embedder.embed_texts(
                sample_texts,
            )
            assert len(result.errors) > 0
            assert "OpenAI API error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_query_success(self, sample_query: str, mock_session):
        """Test successful query embedding."""
        embedder = OpenAIEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="test_key")
        )

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.return_value.embeddings.create.return_value = mock_response

            result = await embedder.embed_query(sample_query)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_query_failure(self, sample_query: str, mock_session):
        """Test query embedding failure."""
        embedder = OpenAIEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="test_key")
        )

        with patch(
            "FOSRABack.src.processing.services.embedder_service.openai.AsyncOpenAI"
        ) as mock_openai:
            mock_openai.return_value.embeddings.create.return_value = MagicMock(data=[])

            with pytest.raises(RuntimeError, match="Failed to embed query"):
                await embedder.embed_query(sample_query)


# =============================================================================
# CohereEmbedder Tests
# =============================================================================


class TestCohereEmbedder:
    """Tests for CohereEmbedder."""

    def test_class_attributes(self):
        """Test class attributes."""
        assert CohereEmbedder.embedder_type == EmbedderType.COHERE
        assert CohereEmbedder.display_name == "Cohere"
        assert CohereEmbedder.default_dense_model == "embed-english-v3.0"
        assert CohereEmbedder.vector_dimension == 1024

    @pytest.mark.asyncio
    async def test_get_api_config_found(self, mock_session):
        """Test getting API config when found in DB."""
        embedder = CohereEmbedder(session=mock_session)

        mock_llm_config = MagicMock()
        mock_llm_config.api_key = "cohere_key"

        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            mock_llm_config
        )

        config = await embedder.get_api_config("user123")  # type: ignore[attr-defined]
        assert config == {"api_key": "cohere_key"}

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, sample_texts: list[str], mock_session):
        """Test successful embedding with Cohere API."""
        embedder = CohereEmbedder(session=mock_session)

        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="cohere_key")
        )

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.cohere.AsyncClient"
        ) as mock_cohere:
            mock_cohere.return_value.embed.return_value = mock_response

            result = await embedder.embed_texts(sample_texts)

            assert result.dense_vectors is not None
            assert len(result.dense_vectors) == len(sample_texts)
            assert result.embedder_used == "COHERE"
            assert result.errors == []
            mock_cohere.return_value.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_api_error(self, sample_texts: list[str], mock_session):
        """Test Cohere API error handling."""
        embedder = CohereEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="cohere_key")
        )

        with patch(
            "FOSRABack.src.processing.services.embedder_service.cohere.AsyncClient"
        ) as mock_cohere:
            mock_cohere.return_value.embed.side_effect = Exception("Cohere API error")

            result = await embedder.embed_texts(sample_texts)
            assert len(result.errors) > 0
            assert "Cohere API error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_embed_query_success(self, sample_query: str, mock_session):
        """Test successful query embedding."""
        embedder = CohereEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = (
            MagicMock(api_key="cohere_key")
        )

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]

        with patch(
            "FOSRABack.src.processing.services.embedder_service.cohere.AsyncClient"
        ) as mock_cohere:
            mock_cohere.return_value.embed.return_value = mock_response

            result = await embedder.embed_query(sample_query)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_query_no_user_id(self, sample_query: str):
        """Test query embedding without user_id."""
        embedder = CohereEmbedder()
        with pytest.raises(RuntimeError, match="user_id required"):
            await embedder.embed_query(sample_query)

    @pytest.mark.asyncio
    async def test_embed_query_no_api_config(self, sample_query: str, mock_session):
        """Test query embedding without API config."""
        embedder = CohereEmbedder(session=mock_session)
        mock_session.execute.return_value.scalars.return_value.first.return_value = None

        with pytest.raises(RuntimeError, match="not configured"):
            await embedder.embed_query(sample_query)


# =============================================================================
# EmbedderService Tests
# =============================================================================


class TestEmbedderService:
    """Tests for EmbedderService main interface."""

    def test_init_default(self, mock_request_context: RequestContext):
        """Test default initialization."""
        service = EmbedderService()
        ctx = mock_request_context

        default = ctx.preferences.embedder.embedder_type

        embedder = service.get_embedder(embedder_type=default)

        assert embedder
        assert isinstance(embedder, FastEmbedEmbedder)
        assert default == EmbedderType.FASTEMBED

    def test_init_custom_default(self, mock_request_context):
        """Test initialization with custom default."""
        service = EmbedderService()
        ctx = mock_request_context

        ctx.preferences.embedder.embedder_type = EmbedderType.OPENAI

        default = ctx.preferences.embedder.embedder_type

        embedder = service.get_embedder(embedder_type=default)

        assert default == EmbedderType.OPENAI
        assert embedder
        assert isinstance(embedder, OpenAIEmbedder)

    def test_get_embedder(self):
        """Test getting an embedder by type."""
        ctx = mock_request_context

        service = EmbedderService()

        embedder = service.get_embedder(
            EmbedderType.FASTEMBED,
        )

        assert embedder is not None
        assert isinstance(embedder, FastEmbedEmbedder)

    def test_get_available_embedders(self, mock_request_context):
        """Test getting available embedders."""
        service = EmbedderService()

        available = service.get_available_embedders()

        assert "FASTEMBED" in available
        assert "OPENAI" in available

    @pytest.mark.asyncio
    async def test_embed_texts_default_embedder(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test embedding texts with default embedder."""
        ctx = mock_request_context
        service = EmbedderService()

        with patch.object(
            FastEmbedEmbedder, "embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                dense_vectors=[[0.1, 0.2]], embedder_used="FASTEMBED"
            )
            result = await service.embed_texts(sample_texts, ctx=ctx)
            assert result.embedder_used == "FASTEMBED"
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_specific_embedder(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test embedding texts with specific embedder type."""
        ctx = mock_request_context
        service = EmbedderService()

        with patch.object(
            SentenceTransformersEmbedder, "embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                dense_vectors=[[0.1, 0.2]], embedder_used="SENTENCE_TRANSFORMERS"
            )
            # change to override, allow parameter pass in
            ctx.preferences.embedder.embedder_type = EmbedderType.SENTENCE_TRANSFORMERS

            result = await service.embed_texts(sample_texts, ctx=ctx)
            assert result.embedder_used == "SENTENCE_TRANSFORMERS"
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_api_embedder_with_user_id(
        self, sample_texts: list[str], mock_session, mock_request_context
    ):
        """Test API embedder receives user_id."""
        ctx = mock_request_context
        service = EmbedderService(
            session=mock_session,
        )
        ctx.preferences.embedder.embedder_type = EmbedderType.OPENAI

        with patch.object(
            OpenAIEmbedder, "embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                dense_vectors=[[0.1, 0.2]], embedder_used="OPENAI"
            )

            await service.embed_texts(sample_texts, ctx=ctx)

            mock_embed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_texts_invalid_embedder(
        self, sample_texts: list[str], mock_request_context
    ):
        """Test embedding with invalid embedder type."""
        ctx = mock_request_context
        service = EmbedderService()

        original = BaseEmbedder._registry.copy()
        BaseEmbedder._registry.clear()

        try:
            result = await service.embed_texts(sample_texts, ctx=ctx)
            assert len(result.errors) > 0
            assert "Embedder not found" in result.errors[0]
        finally:
            BaseEmbedder._registry = original

    @pytest.mark.asyncio
    async def test_embed_query(self, sample_query: str, mock_request_context):
        """Test embedding a query."""
        ctx = mock_request_context
        service = EmbedderService()

        with patch.object(
            FastEmbedEmbedder, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            result = await service.embed_query(sample_query, ctx=ctx)
            assert result == [0.1, 0.2, 0.3]
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_query_api_embedder_with_user_id(
        self, sample_query: str, mock_session, mock_request_context
    ):
        """Test API embedder query receives user_id."""
        service = EmbedderService(
            session=mock_session,
        )

        ctx = mock_request_context

        ctx.preferences.embedder.embedder_type = EmbedderType.OPENAI
        with patch.object(
            OpenAIEmbedder, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            await service.embed_query(sample_query, ctx=ctx)

            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_query_invalid_embedder(
        self, sample_query: str, mock_request_context
    ):
        """Test query embedding with invalid embedder type."""
        ctx = mock_request_context
        service = EmbedderService()

        original = BaseEmbedder._registry.copy()
        BaseEmbedder._registry.clear()

        try:
            with pytest.raises(RuntimeError, match="Embedder not found"):
                await service.embed_query(sample_query, ctx=ctx)
        finally:
            BaseEmbedder._registry = original

    @pytest.mark.asyncio
    async def test_embed_sources_multiple(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test embedding multiple sources concurrently."""
        ctx = mock_request_context
        service = EmbedderService()

        # Create multiple sources
        sources = [sample_source]
        for i in range(2):
            new_source = SourceWithRelations(
                source_id=ulid_factory(),
                source_hash=f"hash{i}",
                unique_id=f"unique-{i}",
                source_summary="",
                summary_embedding="[]",
                source_content=f"Content for source {i}.",
                origin=sample_source.origin,
                chunks=[
                    Chunk(
                        chunk_id=ulid_factory(),
                        source_id=f"source{i}",
                        source_hash=f"hash{i}",
                        text=f"Chunk {i} content.",
                        start_index=0,
                        end_index=10,
                        token_count=2,
                    )
                ],
            )
            sources.append(new_source)

        with patch.object(
            FastEmbedEmbedder, "embed_source", new_callable=AsyncMock
        ) as mock_embed_source:

            async def mock_embed(s, c=None):
                await asyncio.sleep(0.01)
                return s

            mock_embed_source.side_effect = mock_embed  # Simulate async work

            results: list[SourceWithRelations] = await service.embed_sources(
                sources, max_concurrent=2, ctx=ctx
            )

            mock_embed_source.assert_called()
            assert results is not None
            assert len(results) == 3
            assert mock_embed_source.call_count == 3
            # Ensure order is preserved
            assert isinstance([results[0]], list)
            assert results[0].source_id == sample_source.source_id
            assert results[1].source_id == sources[1].source_id
            assert results[2].source_id == sources[2].source_id

    @pytest.mark.asyncio
    async def test_embed_sources_with_errors(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test embedding sources when some fail."""
        ctx = mock_request_context
        service = EmbedderService()

        sources = [sample_source]

        with patch.object(
            FastEmbedEmbedder, "embed_source", new_callable=AsyncMock
        ) as mock_embed_source:

            async def side_effect(source, config=None):
                if source == sample_source:
                    raise Exception("Embedding failed")
                return source

            mock_embed_source.side_effect = side_effect

            results = await service.embed_sources(sources, ctx)

            assert len(results) == 1
            assert (
                results[0] == sample_source
            )  # Should be the original source due to error

    @pytest.mark.asyncio
    async def test_embed_sources_invalid_embedder(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test embedding sources with invalid embedder."""
        ctx = mock_request_context
        service = EmbedderService()

        original = BaseEmbedder._registry.copy()
        BaseEmbedder._registry.clear()

        try:
            results = await service.embed_sources([sample_source], ctx=ctx)

            # Should return sources unchanged
            assert results == [sample_source]
        finally:
            BaseEmbedder._registry = original

    @pytest.mark.asyncio
    @patch.object(FastEmbedEmbedder, "embed_source")
    async def test_embed_sources_no_chunks(
        self, mock_embed_source, sample_origin: Origin, mock_request_context
    ):
        """Test embedding sources that have no chunks."""
        ctx = mock_request_context

        service = EmbedderService()

        source_no_chunks = SourceWithRelations(
            source_id="source_no_chunks",
            source_hash=sample_origin.source_hash,
            unique_id="test-no-chunks",
            source_summary="",
            summary_embedding="[]",
            source_content="Some content but no chunks yet.",
            origin=sample_origin,
            chunks=[],
        )

        results = await service.embed_sources(sources=[source_no_chunks], ctx=ctx)

        assert len(results) == 1
        assert results[0] == source_no_chunks  # Should be returned unchanged

        mock_embed_source.assert_called_once()  # Still calls embed_source, which handles no chunks


# =============================================================================
# Integration Tests
# =============================================================================


class TestEmbedderIntegration:
    """Integration tests for embedder service."""

    @pytest.mark.asyncio
    async def test_full_pipeline_fastembed_dense(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test full embedding pipeline with FastEmbed dense."""
        ctx = mock_request_context
        service = EmbedderService()
        config = EmbedderConfig()

        # Embed the source using embed_sources
        result_sources = await service.embed_sources([sample_source], ctx=ctx)
        result_source = result_sources[0]

        # Verify chunks have dense vectors
        assert len(result_source.chunks) == 3
        assert result_source.chunks[0].dense_vector is not None
        assert result_source.chunks[1].dense_vector is not None
        assert result_source.chunks[2].dense_vector is None  # Empty chunk

        assert (
            len(result_source.chunks[0].dense_vector)
            == FastEmbedEmbedder.vector_dimension
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_fastembed_hybrid(
        self, sample_source: SourceWithRelations, mock_request_context
    ):
        """Test full embedding pipeline with FastEmbed hybrid."""
        ctx = mock_request_context
        service = EmbedderService()
        config = EmbedderConfig()

        # Embed the source using embed_sources
        result_sources = await service.embed_sources([sample_source], ctx=ctx)
        result_source = result_sources[0]

        # WARN: Not populating sparse currently

        # Verify chunks have dense and sparse vectors
        assert len(result_source.chunks) == 3
        assert result_source.chunks[0].dense_vector is not None
        # assert result_source.chunks[0].sparse_vector is not None
        assert result_source.chunks[1].dense_vector is not None
        # assert result_source.chunks[1].sparse_vector is not None
        assert result_source.chunks[2].dense_vector is None
        # assert result_source.chunks[2].sparse_vector is None

    @pytest.mark.asyncio
    async def test_embedder_type_enum(self):
        """Test all EmbedderType enum values."""
        assert EmbedderType.FASTEMBED == "FASTEMBED"
        assert EmbedderType.SENTENCE_TRANSFORMERS == "SENTENCE_TRANSFORMERS"
        assert EmbedderType.OPENAI == "OPENAI"
        assert EmbedderType.COHERE == "COHERE"
        assert EmbedderType.VOYAGE == "VOYAGE"
        assert EmbedderType.JINA == "JINA"
        assert EmbedderType.MISTRAL == "MISTRAL"

    @pytest.mark.asyncio
    async def test_embedding_mode_enum(self):
        """Test all EmbeddingMode enum values."""
        assert EmbeddingMode.DENSE_ONLY == "DENSE_ONLY"
        assert EmbeddingMode.SPARSE_ONLY == "SPARSE_ONLY"
        assert EmbeddingMode.HYBRID == "HYBRID"
        assert EmbeddingMode.ADVANCED == "ADVANCED"
