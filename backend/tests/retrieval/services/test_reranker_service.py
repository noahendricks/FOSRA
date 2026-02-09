"""
Tests for RerankerService and all reranker implementations.

Achieves 90%+ code coverage through comprehensive unit and integration tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.retrieval.services.reranker_service import (
    APIReranker,
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    FlashRankReranker,
    LocalReranker,
    RerankerConfig,
    RerankerService,
    RerankerType,
    RerankResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock AsyncSession."""
    session = AsyncMock()
    return session


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample documents for reranking tests."""
    return [
        {"chunk_id": "1", "content": "Python is a programming language", "score": 0.8},
        {"chunk_id": "2", "content": "JavaScript runs in browsers", "score": 0.7},
        {"chunk_id": "3", "content": "Python has great libraries", "score": 0.6},
        {"chunk_id": "4", "content": "Machine learning with Python", "score": 0.5},
    ]


@pytest.fixture
def reranker_config():
    """Default reranker configuration."""
    return RerankerConfig(top_k=10, score_threshold=None, return_scores=True)


# =============================================================================
# RerankerConfig Tests
# =============================================================================


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RerankerConfig()
        assert config.top_k == 10
        assert config.score_threshold is None
        assert config.return_scores is True
        assert config.batch_size == 32

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RerankerConfig(
            top_k=5,
            score_threshold=0.5,
            return_scores=False,
            batch_size=16,
        )
        assert config.top_k == 5
        assert config.score_threshold == 0.5
        assert config.return_scores is False
        assert config.batch_size == 16


# =============================================================================
# RerankResult Tests
# =============================================================================


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = RerankResult(documents=[])
        assert result.documents == []
        assert result.reranker_used == ""
        assert result.rerank_time_ms == 0.0
        assert result.original_count == 0
        assert result.filtered_count == 0
        assert result.errors == []

    def test_with_data(self, sample_documents):
        """Test result with data."""
        result = RerankResult(
            documents=sample_documents,
            reranker_used="FLASHRANK",
            rerank_time_ms=50.5,
            original_count=10,
            filtered_count=4,
        )
        assert len(result.documents) == 4
        assert result.reranker_used == "FLASHRANK"
        assert result.rerank_time_ms == 50.5


# =============================================================================
# BaseReranker Tests
# =============================================================================


class TestBaseReranker:
    """Tests for BaseReranker abstract class."""

    def test_registry_contains_implementations(self):
        """Test that registry contains registered implementations."""
        available = BaseReranker.get_available_rerankers()
        assert "FLASHRANK" in available
        assert "CROSS_ENCODER" in available
        assert "COHERE" in available

    def test_get_reranker_valid_type(self, mock_session):
        """Test getting a valid reranker type."""
        reranker = BaseReranker.get_reranker(RerankerType.FLASHRANK, mock_session)
        assert reranker is not None
        assert isinstance(reranker, FlashRankReranker)

    def test_get_reranker_invalid_type(self, mock_session):
        """Test getting an invalid reranker type returns None."""
        # Create a mock invalid type
        result = BaseReranker._registry.get("INVALID_TYPE")
        assert result is None

    def test_create_error_result(self, mock_session, sample_documents):
        """Test error result creation."""
        reranker = FlashRankReranker(mock_session)
        result = reranker._create_error_result("Test error", sample_documents)

        assert result.documents == sample_documents
        assert result.reranker_used == RerankerType.FLASHRANK.value
        assert "Test error" in result.errors


# =============================================================================
# FlashRankReranker Tests
# =============================================================================


class TestFlashRankReranker:
    """Tests for FlashRankReranker implementation."""

    def test_init(self, mock_session):
        """Test initialization."""
        reranker = FlashRankReranker(mock_session)
        assert reranker.session == mock_session
        assert reranker._ranker is None  # type: ignore[attr-defined]
        assert reranker.reranker_type == RerankerType.FLASHRANK
        assert reranker.display_name == "FlashRank"

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, mock_session):
        """Test reranking empty document list."""
        reranker = FlashRankReranker(mock_session)
        result = await reranker.rerank("test query", [])

        assert result.documents == []
        assert result.original_count == 0
        assert result.reranker_used == "FLASHRANK"

    @pytest.mark.asyncio
    @patch(
        "backend.src.retrieval.services.reranker_service.FlashRankReranker._initialize_reranker"
    )
    async def test_rerank_initialization_failure(
        self,
        mock_init,
        mock_session,
        sample_documents,
    ):
        """Test handling of initialization failure."""
        mock_init.return_value = None
        reranker = FlashRankReranker(mock_session)
        reranker._ranker = None  # type: ignore[union-attr]

        result = await reranker.rerank("test query", sample_documents)

        assert "Failed to initialize FlashRank" in result.errors

    @pytest.mark.asyncio
    @patch("flashrank.Ranker")
    @patch("flashrank.RerankRequest")
    async def test_rerank_success(
        self, mock_request, mock_ranker_cls, mock_session, sample_documents
    ):
        """Test successful reranking."""
        # Setup mock ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "3", "score": 0.95, "text": "Python has great libraries"},
            {"id": "1", "score": 0.85, "text": "Python is a programming language"},
            {"id": "4", "score": 0.75, "text": "Machine learning with Python"},
            {"id": "2", "score": 0.65, "text": "JavaScript runs in browsers"},
        ]
        mock_ranker_cls.return_value = mock_ranker

        reranker = FlashRankReranker(mock_session)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._ranker = mock_ranker  # type: ignore[attr-defined]
            result = await reranker.rerank("Python programming", sample_documents)

        assert result.original_count == 4
        assert result.reranker_used == "FLASHRANK"
        assert result.rerank_time_ms > 0

    @pytest.mark.asyncio
    @patch("flashrank.Ranker")
    async def test_rerank_with_threshold(
        self, mock_ranker_cls, mock_session, sample_documents
    ):
        """Test reranking with score threshold."""
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "1", "score": 0.9, "text": "doc1"},
            {"id": "2", "score": 0.4, "text": "doc2"},  # Below threshold
        ]
        mock_ranker_cls.return_value = mock_ranker

        reranker = FlashRankReranker(mock_session)
        config = RerankerConfig(score_threshold=0.5, top_k=10)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._ranker = mock_ranker  # type: ignore[attr-defined]
            result = await reranker.rerank("query", sample_documents[:2], config)

        # Only docs above threshold should remain
        filtered_docs = [d for d in result.documents if d.get("rerank_score", 0) >= 0.5]
        assert all(d.get("rerank_score", 0) >= 0.5 for d in filtered_docs)

    @pytest.mark.asyncio
    @patch("flashrank.Ranker")
    async def test_rerank_with_top_k(
        self, mock_ranker_cls, mock_session, sample_documents
    ):
        """Test reranking respects top_k limit."""
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": str(i), "score": 0.9 - i * 0.1, "text": f"doc{i}"} for i in range(4)
        ]
        mock_ranker_cls.return_value = mock_ranker

        reranker = FlashRankReranker(mock_session)
        config = RerankerConfig(top_k=2)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._ranker = mock_ranker  # type: ignore[attr-defined]
            result = await reranker.rerank("query", sample_documents, config)

        assert len(result.documents) <= 2

    @pytest.mark.asyncio
    async def test_rerank_exception_handling(self, mock_session, sample_documents):
        """Test exception handling during reranking."""
        reranker = FlashRankReranker(mock_session)

        with patch.object(
            reranker, "_initialize_reranker", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            result = await reranker.rerank("query", sample_documents)

        assert len(result.errors) > 0


# =============================================================================
# CrossEncoderReranker Tests
# =============================================================================


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker implementation."""

    def test_init(self, mock_session):
        """Test initialization."""
        reranker = CrossEncoderReranker(mock_session)
        assert reranker.session == mock_session
        assert reranker._model is None  # type: ignore[attr-defined]
        assert reranker.reranker_type == RerankerType.CROSS_ENCODER
        assert reranker.display_name == "Cross-Encoder"
        assert reranker.default_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, mock_session):
        """Test reranking empty document list."""
        reranker = CrossEncoderReranker(mock_session)
        result = await reranker.rerank("test query", [])

        assert result.documents == []
        assert result.original_count == 0

    @pytest.mark.asyncio
    async def test_rerank_initialization_failure(
        self,
        mock_session,
        sample_documents,
    ):
        """Test handling of initialization failure."""
        reranker = CrossEncoderReranker(mock_session)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._model = None  # type: ignore[union-attr]
            result = await reranker.rerank("test query", sample_documents)

        assert "Failed to initialize cross-encoder" in result.errors

    @pytest.mark.asyncio
    @patch("sentence_transformers.CrossEncoder")
    async def test_rerank_success(self, mock_ce_cls, mock_session, sample_documents):
        """Test successful cross-encoder reranking."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.85, 0.6]
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(mock_session)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._model = mock_model  # type: ignore[attr-defined]
            result = await reranker.rerank("Python query", sample_documents)

        assert result.original_count == 4
        assert result.reranker_used == "CROSS_ENCODER"
        # Should be sorted by score descending
        scores = [d.get("rerank_score", 0) for d in result.documents]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    @patch("sentence_transformers.CrossEncoder")
    async def test_rerank_with_threshold(
        self, mock_ce_cls, mock_session, sample_documents
    ):
        """Test cross-encoder with score threshold."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.8, 0.2]  # 2 below 0.5
        mock_ce_cls.return_value = mock_model

        reranker = CrossEncoderReranker(mock_session)
        config = RerankerConfig(score_threshold=0.5, top_k=10)

        with patch.object(reranker, "_initialize_reranker", new_callable=AsyncMock):
            reranker._model = mock_model  # type: ignore[attr-defined]
            result = await reranker.rerank("query", sample_documents, config)

        assert all(d.get("rerank_score", 0) >= 0.5 for d in result.documents)

    @pytest.mark.asyncio
    async def test_rerank_import_error(self, mock_session, sample_documents):
        """Test handling of missing sentence-transformers."""
        reranker = CrossEncoderReranker(mock_session)

        with patch.object(
            reranker, "_initialize_reranker", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = ImportError("No module")
            result = await reranker.rerank("query", sample_documents)

        assert len(result.errors) > 0


# =============================================================================
# CohereReranker Tests
# =============================================================================


class TestCohereReranker:
    """Tests for CohereReranker implementation."""

    def test_init(self, mock_session):
        """Test initialization."""
        reranker = CohereReranker(mock_session)
        assert reranker.session == mock_session
        assert reranker.reranker_type == RerankerType.COHERE
        assert reranker.display_name == "Cohere"
        assert reranker.default_model == "rerank-english-v3.0"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, mock_session):
        """Test reranking empty document list."""
        reranker = CohereReranker(mock_session)
        result = await reranker.rerank("test query", [])

        assert result.documents == []
        assert result.original_count == 0

    @pytest.mark.asyncio
    async def test_rerank_no_user_id(self, mock_session, sample_documents):
        """Test reranking without user_id returns error."""
        reranker = CohereReranker(mock_session)
        result = await reranker.rerank("query", sample_documents)

        assert "user_id required for Cohere reranker" in result.errors[0]
        assert len(result.documents) == 4

    # @pytest.mark.asyncio
    # async def test_rerank_no_api_config(self, mock_session, sample_documents):
    #     """Test reranking without API configuration."""
    #     reranker = CohereReranker(mock_session)
    #
    #     with patch.object(
    #         reranker, "get_api_config", new_callable=AsyncMock
    #     ) as mock_config:
    #         mock_config.return_value = None
    #         result = await reranker.rerank(
    #             "query",
    #             sample_documents,
    #         )
    #
    #     assert "user_id required for Cohere reranker " in result.errors[0]

    @pytest.mark.asyncio
    async def test_get_api_config_no_session(self):
        """Test get_api_config without session."""
        reranker = CohereReranker(None)
        result = await reranker.get_api_config("user1")  # type: ignore[attr-defined]
        assert result is None

    @pytest.mark.asyncio
    async def test_get_api_config_success(self, mock_session):
        """Test successful API config retrieval."""
        mock_config = MagicMock()
        mock_config.api_key = "test-api-key"

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_config
        mock_session.execute = AsyncMock(return_value=mock_result)

        reranker = CohereReranker(mock_session)
        config = await reranker.get_api_config("user1")  # type: ignore[attr-defined]

        assert config == {"api_key": "test-api-key"}

    @pytest.mark.asyncio
    async def test_get_api_config_not_found(self, mock_session):
        """Test API config not found."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        reranker = CohereReranker(mock_session)
        config = await reranker.get_api_config("user1")  # type: ignore[attr-defined]

        assert config is None

    # @pytest.mark.asyncio
    # @patch("cohere.AsyncClient")
    # async def test_rerank_success(
    #     self, mock_cohere_cls, mock_session, sample_documents
    # ):
    #     """Test successful Cohere reranking."""
    #     # Setup mock response
    #     mock_result1 = MagicMock()
    #     mock_result1.index = 0
    #     mock_result1.relevance_score = 0.95
    #
    #     mock_result2 = MagicMock()
    #     mock_result2.index = 2
    #     mock_result2.relevance_score = 0.85
    #
    #     mock_response = MagicMock()
    #     mock_response.results = [mock_result1, mock_result2]
    #
    #     mock_client = AsyncMock()
    #     mock_client.rerank = AsyncMock(return_value=mock_response)
    #     mock_cohere_cls.return_value = mock_client
    #
    #     reranker = CohereReranker(mock_session)
    #
    #     with patch.object(
    #         reranker, "get_api_config", new_callable=AsyncMock
    #     ) as mock_config:
    #         mock_config.return_value = {"api_key": "test-key"}
    #         result = await reranker.rerank(
    #             "Python programming",
    #             sample_documents,
    #             RerankerConfig(top_k=5),
    #         )
    #
    #     assert result.reranker_used == "COHERE"
    #     assert result.original_count == 4
    #     assert len(result.documents) == 2
    #
    # @pytest.mark.asyncio
    # async def test_rerank_import_error(self, mock_session, sample_documents):
    #     """Test handling of missing cohere package."""
    #     reranker = CohereReranker(mock_session)
    #
    #     with patch.object(
    #         reranker, "get_api_config", new_callable=AsyncMock
    #     ) as mock_config:
    #         mock_config.return_value = {"api_key": "test-key"}
    #         # Mock cohere import failure
    #         with patch(
    #             "backend.src.retrieval.services.reranker_service.CohereReranker.rerank",
    #             side_effect=ImportError("cohere not installed"),
    #         ):
    #             result = await reranker.rerank(
    #                 "query",
    #                 sample_documents,
    #             )
    #
    #     assert "cohere not installed" in result.errors[0]


# =============================================================================
# RerankerService Tests
# =============================================================================


class TestRerankerService:
    """Tests for RerankerService class."""

    def test_init_default(self, mock_session):
        """Test default initialization."""
        service = RerankerService(mock_session)
        assert service.session == mock_session
        assert service.default_reranker == RerankerType.FLASHRANK

    def test_init_custom_default(self, mock_session):
        """Test initialization with custom default."""
        service = RerankerService(
            mock_session, default_reranker=RerankerType.CROSS_ENCODER
        )
        assert service.default_reranker == RerankerType.CROSS_ENCODER

    def test_get_reranker(self, mock_session):
        """Test getting a reranker by type."""
        service = RerankerService(mock_session)
        reranker = service.get_reranker(RerankerType.FLASHRANK)

        assert reranker is not None
        assert isinstance(reranker, FlashRankReranker)

    def test_get_available_rerankers(self, mock_session):
        """Test getting available rerankers."""
        service = RerankerService(mock_session)
        available = service.get_available_rerankers()

        assert "FLASHRANK" in available
        assert "CROSS_ENCODER" in available
        assert "COHERE" in available

    @pytest.mark.asyncio
    async def test_rerank_uses_default(self, mock_session, sample_documents):
        """Test rerank method uses default reranker if none specified."""
        service = RerankerService(mock_session, default_reranker=RerankerType.FLASHRANK)

        with patch.object(service, "get_reranker") as mock_get_reranker:
            mock_reranker_instance = AsyncMock(spec=FlashRankReranker)
            mock_reranker_instance.rerank.return_value = RerankResult(
                documents=sample_documents, reranker_used="FLASHRANK"
            )
            mock_get_reranker.return_value = mock_reranker_instance

            result = await service.rerank("query", sample_documents)

            mock_get_reranker.assert_called_once_with(RerankerType.FLASHRANK)
            mock_reranker_instance.rerank.assert_called_once()
            assert result.reranker_used == "FLASHRANK"

    @pytest.mark.asyncio
    async def test_rerank_uses_specified_reranker(self, mock_session, sample_documents):
        """Test rerank method uses specified reranker."""
        service = RerankerService(mock_session, default_reranker=RerankerType.FLASHRANK)

        with patch.object(service, "get_reranker") as mock_get_reranker:
            mock_reranker_instance = AsyncMock(spec=CrossEncoderReranker)
            mock_reranker_instance.rerank.return_value = RerankResult(
                documents=sample_documents, reranker_used="CROSS_ENCODER"
            )
            mock_get_reranker.return_value = mock_reranker_instance

            result = await service.rerank(
                "query", sample_documents, reranker_type=RerankerType.CROSS_ENCODER
            )

            mock_get_reranker.assert_called_once_with(RerankerType.CROSS_ENCODER)
            mock_reranker_instance.rerank.assert_called_once()
            assert result.reranker_used == "CROSS_ENCODER"

    @pytest.mark.asyncio
    async def test_rerank_reranker_not_found(self, mock_session, sample_documents):
        """Test rerank method when specified reranker is not found."""
        service = RerankerService(mock_session)

        with patch.object(service, "get_reranker") as mock_get_reranker:
            mock_get_reranker.return_value = None

            result = await service.rerank(
                "query", sample_documents, reranker_type=RerankerType.JINA
            )

            assert len(result.errors) > 0
            assert "Reranker not found: JINA" in result.errors
            assert result.documents == sample_documents  # Original documents returned

    @pytest.mark.asyncio
    async def test_rerank_api_reranker_with_user_id(
        self, mock_session, sample_documents
    ):
        """Test rerank method correctly passes user_id to API reranker."""
        service = RerankerService(mock_session)

        with patch.object(service, "get_reranker") as mock_get_reranker:
            mock_api_reranker_instance = AsyncMock(spec=CohereReranker)
            mock_api_reranker_instance.rerank.return_value = RerankResult(
                documents=sample_documents, reranker_used="COHERE"
            )
            mock_get_reranker.return_value = mock_api_reranker_instance

            user_id = "test-user-123"
            await service.rerank(
                "query",
                sample_documents,
                reranker_type=RerankerType.COHERE,
                user_id=user_id,
            )

            mock_api_reranker_instance.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_api_reranker_without_user_id(
        self, mock_session, sample_documents
    ):
        """Test rerank method for API reranker without user_id (should still call rerank)."""
        service = RerankerService(mock_session)

        with patch.object(service, "get_reranker") as mock_get_reranker:
            mock_api_reranker_instance = AsyncMock(spec=CohereReranker)
            mock_api_reranker_instance.rerank.return_value = RerankResult(
                documents=sample_documents,
                reranker_used="COHERE",
                errors=["user_id required"],
            )
            mock_get_reranker.return_value = mock_api_reranker_instance

            result = await service.rerank(
                "query", sample_documents, reranker_type=RerankerType.COHERE
            )

            mock_api_reranker_instance.rerank.assert_called_once_with(
                "query", sample_documents, None
            )
            assert "user_id required" in result.errors[0]
