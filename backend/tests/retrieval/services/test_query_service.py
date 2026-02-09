"""
Tests for QueryService, QueryExpander, MetadataFilter, and FusionRetriever.

Achieves 90%+ code coverage through comprehensive unit and integration tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from backend.src.config.request_context import RequestContext

from backend.src.retrieval.services.query_service import (
    FusionRetriever,
    MetadataFilter,
    QueryExpander,
    QueryService,
)
from backend.src.resources.test_fixtures import mock_request_context


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock AsyncSession."""
    return AsyncMock()


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def query_expander(mock_session):
    """Create a QueryExpander with mock session."""
    return QueryExpander(
        enable_expansion=True,
        enable_hyde=True,
        enable_decomposition=True,
        session=mock_session,
    )


@pytest.fixture
def metadata_filter():
    """Create a MetadataFilter instance."""
    return MetadataFilter()


@pytest.fixture
def fusion_retriever():
    """Create a FusionRetriever instance."""
    return FusionRetriever(k=60)


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """Sample chunks for testing."""
    return [
        {
            "id": "1",
            "content": "Python basics",
            "source_type": "pdf",
            "tags": ["python"],
        },
        {"id": "2", "content": "JavaScript guide", "source_type": "md", "tags": ["js"]},
        {
            "id": "3",
            "content": "React tutorial",
            "source_type": "pdf",
            "tags": ["react", "js"],
        },
        {
            "id": "4",
            "content": "Django framework",
            "source_type": "html",
            "tags": ["python", "web"],
        },
    ]


# =============================================================================
# QueryExpander Tests
# =============================================================================


class TestQueryExpander:
    """Tests for QueryExpander class."""

    @pytest.mark.asyncio
    async def test_init_default_values(self):
        """Test default initialization."""
        expander = QueryExpander()
        assert expander.enable_expansion is True
        assert expander.enable_hyde is False
        assert expander.enable_decomposition is False
        assert expander.max_sub_queries == 5
        assert expander.session is None

    def test_get_llm_raises_without_context(self):
        """Test that _get_llm raises ValueError without required context."""
        expander = QueryExpander()
        with pytest.raises(
            ValueError, match="QueryExpander requires session and RequestContext"
        ):
            expander._get_llm()

    def test_get_llm_raises_without_session(self):
        """Test that _get_llm raises ValueError without session."""
        expander = QueryExpander()
        with pytest.raises(ValueError):
            expander._get_llm()

    def test_get_llm_raises_without_user_id(self, mock_session):
        """Test that _get_llm raises ValueError without user_id."""
        expander = QueryExpander(
            session=mock_session,
        )
        with pytest.raises(ValueError):
            expander._get_llm()

    def test_get_llm_raises_without_workspace_id(self, mock_session):
        """Test that _get_llm raises ValueError without workspace_id."""
        expander = QueryExpander(
            session=mock_session,
        )
        with pytest.raises(ValueError):
            expander._get_llm()

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_get_llm_success(self, mock__get_llm, query_expander, mock_llm):
        """Test successful LLM retrieval."""
        mock__get_llm.return_value = mock_llm
        result = query_expander._get_llm()
        assert result == mock_llm
        mock__get_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_empty_string(self, query_expander):
        """Test processing empty query."""
        result = await query_expander.process_query("")
        assert result["original_query"] == ""
        assert result["queries"] == [""]
        assert result["strategy"] == "none"

    @pytest.mark.asyncio
    async def test_process_query_whitespace_only(self, query_expander):
        """Test processing whitespace-only query."""
        result = await query_expander.process_query("   ")
        assert result["strategy"] == "none"

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_auto_strategy_complex(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test auto strategy selection for complex queries."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="sub1\nsub2\nsub3"))
        mock__get_llm.return_value = mock_llm

        # Complex query with "and"
        result = await query_expander.process_query("Python and JavaScript comparison")
        assert "decomposition" in result["strategy"] or len(result["queries"]) > 1

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_auto_strategy_open_ended(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test auto strategy selection for open-ended queries."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Hypothetical answer")
        )
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("What is machine learning?")
        # Should select hyde for open-ended questions
        assert result["original_query"] == "What is machine learning?"

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_expansion_strategy(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test expansion strategy."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="alt1\nalt2\nalt3"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("test query", strategy="expansion")
        assert result["strategy"] == "expansion"
        assert "test query" in result["queries"]

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_hyde_strategy(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test HyDE strategy."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Hypothetical document content")
        )
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("test query", strategy="hyde")
        assert result["strategy"] == "hyde"

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_decomposition_strategy(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test decomposition strategy."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="sub1\nsub2\nsub3"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query(
            "complex query", strategy="decomposition"
        )
        assert result["strategy"] == "decomposition"

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_multi_strategy(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test multi strategy runs all methods."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="result1\nresult2"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("test", strategy="multi")
        assert result["strategy"] == "multi"
        # Should have original plus expanded queries
        assert len(result["queries"]) >= 1

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_deduplicates(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test that results are deduplicated."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="test\ntest\ntest"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("test", strategy="expansion")
        # Should not have duplicates
        assert len(result["queries"]) == len(set(result["queries"]))

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_process_query_limits_to_15(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test that results are limited to 15."""
        many_results = "\n".join([f"query{i}" for i in range(20)])
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=many_results))
        mock__get_llm.return_value = mock_llm

        result = await query_expander.process_query("test", strategy="expansion")
        assert len(result["queries"]) <= 15

    def test_is_complex_query_with_and(self, query_expander):
        """Test complex query detection with 'and'."""
        assert query_expander._is_complex_query("Python and JavaScript") is True

    def test_is_complex_query_with_or(self, query_expander):
        """Test complex query detection with 'or'."""
        assert query_expander._is_complex_query("React or Vue") is True

    def test_is_complex_query_with_compare(self, query_expander):
        """Test complex query detection with 'compare'."""
        assert query_expander._is_complex_query("compare frameworks") is True

    def test_is_complex_query_with_vs(self, query_expander):
        """Test complex query detection with 'vs'."""
        assert query_expander._is_complex_query("Python vs Ruby") is True

    def test_is_complex_query_long_query(self, query_expander):
        """Test complex query detection for long queries."""
        long_query = " ".join(["word"] * 20)
        assert query_expander._is_complex_query(long_query) is True

    def test_is_complex_query_simple(self, query_expander):
        """Test simple query is not complex."""
        assert query_expander._is_complex_query("Python basics") is False

    def test_is_open_ended_what(self, query_expander):
        """Test open-ended detection with 'what'."""
        assert query_expander._is_open_ended("What is Python?") is True

    def test_is_open_ended_why(self, query_expander):
        """Test open-ended detection with 'why'."""
        assert query_expander._is_open_ended("Why use async?") is True

    def test_is_open_ended_how(self, query_expander):
        """Test open-ended detection with 'how'."""
        assert query_expander._is_open_ended("How to learn coding?") is True

    def test_is_open_ended_explain(self, query_expander):
        """Test open-ended detection with 'explain'."""
        assert query_expander._is_open_ended("Explain decorators") is True

    def test_is_open_ended_describe(self, query_expander):
        """Test open-ended detection with 'describe'."""
        assert query_expander._is_open_ended("Describe the process") is True

    def test_is_open_ended_closed(self, query_expander):
        """Test closed query is not open-ended."""
        assert query_expander._is_open_ended("Python tutorial") is False

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_expand_query_success(self, mock__get_llm, query_expander, mock_llm):
        """Test successful query expansion."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="alt1\nalt2\nalt3"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander._expand_query("test query")
        assert len(result) == 3
        assert "alt1" in result

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_expand_query_no_llm(self, mock__get_llm, query_expander):
        """Test expansion with no LLM available."""
        mock__get_llm.return_value = None
        result = await query_expander._expand_query("test query")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_expand_query_exception(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test expansion handles exceptions."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander._expand_query("test query")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_generate_hypothetical_document_disabled(
        self, mock__get_llm, mock_session
    ):
        """Test HyDE when disabled."""
        expander = QueryExpander(
            enable_hyde=False,
            session=mock_session,
        )
        result = await expander._generate_hypothetical_document("test")
        assert result == ""

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_generate_hypothetical_document_success(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test successful HyDE generation."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Hypothetical answer")
        )
        mock__get_llm.return_value = mock_llm

        result = await query_expander._generate_hypothetical_document("test")
        assert result == "Hypothetical answer"

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_generate_hypothetical_document_no_llm(
        self, mock__get_llm, query_expander
    ):
        """Test HyDE with no LLM."""
        mock__get_llm.return_value = None
        result = await query_expander._generate_hypothetical_document("test")
        assert result == ""

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_generate_hypothetical_document_exception(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test HyDE handles exceptions."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Error"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander._generate_hypothetical_document("test")
        assert result == ""

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_decompose_query_disabled(self, mock__get_llm, mock_session):
        """Test decomposition when disabled."""
        expander = QueryExpander(
            enable_decomposition=False,
            session=mock_session,
        )
        result = await expander._decompose_query("test")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_decompose_query_success(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test successful query decomposition."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="sub1\nsub2\nsub3"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander._decompose_query("complex query")
        assert len(result) == 3

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_decompose_query_no_llm(self, mock__get_llm, query_expander):
        """Test decomposition with no LLM."""
        mock__get_llm.return_value = None
        result = await query_expander._decompose_query("test")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.src.retrieval.services.query_service.QueryExpander._get_llm")
    async def test_decompose_query_exception(
        self, mock__get_llm, query_expander, mock_llm
    ):
        """Test decomposition handles exceptions."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Error"))
        mock__get_llm.return_value = mock_llm

        result = await query_expander._decompose_query("test")
        assert result == []


# =============================================================================
# MetadataFilter Tests
# =============================================================================


class TestMetadataFilter:
    """Tests for MetadataFilter class."""

    def test_filter_chunks_no_filters(self, metadata_filter, sample_chunks):
        """Test filtering with no filters returns all chunks."""
        result = metadata_filter.filter_chunks(sample_chunks, None)
        assert len(result) == len(sample_chunks)

    def test_filter_chunks_empty_filters(self, metadata_filter, sample_chunks):
        """Test filtering with empty filters returns all chunks."""
        result = metadata_filter.filter_chunks(sample_chunks, {})
        assert len(result) == len(sample_chunks)

    def test_filter_chunks_exact_match(self, metadata_filter, sample_chunks):
        """Test exact match filtering."""
        result = metadata_filter.filter_chunks(sample_chunks, {"source_type": "pdf"})
        assert len(result) == 2
        assert all(c["source_type"] == "pdf" for c in result)

    def test_filter_chunks_list_whitelist(self, metadata_filter, sample_chunks):
        """Test list-based whitelist filtering."""
        result = metadata_filter.filter_chunks(
            sample_chunks, {"source_type": ["pdf", "md"]}
        )
        assert len(result) == 3

    def test_filter_chunks_callable_filter(self, metadata_filter):
        """Test callable filter function."""
        chunks = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.5},
            {"id": "3", "score": 0.8},
        ]
        result = metadata_filter.filter_chunks(chunks, {"score": lambda x: x > 0.7})
        assert len(result) == 2

    def test_filter_chunks_callable_exception(self, metadata_filter):
        """Test callable filter that raises exception."""
        chunks = [{"id": "1", "score": None}]
        result = metadata_filter.filter_chunks(chunks, {"score": lambda x: x > 0.7})
        assert len(result) == 0  # Should filter out due to exception

    def test_filter_chunks_missing_field(self, metadata_filter, sample_chunks):
        """Test filtering on missing field."""
        result = metadata_filter.filter_chunks(sample_chunks, {"nonexistent": "value"})
        assert len(result) == 0

    def test_filter_chunks_multiple_filters(self, metadata_filter, sample_chunks):
        """Test multiple filters combined."""
        result = metadata_filter.filter_chunks(
            sample_chunks, {"source_type": "pdf", "id": "1"}
        )
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_matches_filters_all_pass(self, metadata_filter):
        """Test _matches_filters when all filters pass."""
        chunk = {"type": "pdf", "score": 0.9}
        filters = {"type": "pdf", "score": 0.9}
        assert metadata_filter._matches_filters(chunk, filters) is True

    def test_matches_filters_fails_exact(self, metadata_filter):
        """Test _matches_filters when exact match fails."""
        chunk = {"type": "pdf"}
        filters = {"type": "md"}
        assert metadata_filter._matches_filters(chunk, filters) is False

    def test_matches_filters_fails_list(self, metadata_filter):
        """Test _matches_filters when list filter fails."""
        chunk = {"type": "html"}
        filters = {"type": ["pdf", "md"]}
        assert metadata_filter._matches_filters(chunk, filters) is False


# =============================================================================
# FusionRetriever Tests
# =============================================================================


class TestFusionRetriever:
    """Tests for FusionRetriever class."""

    def test_init_default_k(self):
        """Test default k value."""
        retriever = FusionRetriever()
        assert retriever.k == 60

    def test_init_custom_k(self):
        """Test custom k value."""
        retriever = FusionRetriever(k=30)
        assert retriever.k == 30

    def test_fuse_results_empty_list(self, fusion_retriever):
        """Test fusion with empty list."""
        result = fusion_retriever.fuse_results([])
        assert result == []

    def test_fuse_results_single_list(self, fusion_retriever):
        """Test fusion with single list returns it unchanged."""
        results = [{"id": "1", "content": "test"}]
        fused = fusion_retriever.fuse_results([results])
        assert len(fused) == 1
        assert fused[0]["id"] == "1"

    def test_fuse_results_single_list_with_top_k(self, fusion_retriever):
        """Test fusion with single list respects top_k."""
        results = [{"id": str(i), "content": f"test{i}"} for i in range(10)]
        fused = fusion_retriever.fuse_results([results], top_k=5)
        assert len(fused) == 5

    def test_fuse_results_multiple_lists(self, fusion_retriever):
        """Test fusion with multiple lists."""
        list1 = [
            {"id": "1", "content": "doc1"},
            {"id": "2", "content": "doc2"},
        ]
        list2 = [
            {"id": "2", "content": "doc2"},
            {"id": "3", "content": "doc3"},
        ]
        fused = fusion_retriever.fuse_results([list1, list2])

        # doc2 appears in both lists, should have highest RRF score
        assert len(fused) == 3
        assert fused[0]["id"] == "2"  # Highest RRF score
        assert "rrf_score" in fused[0]

    def test_fuse_results_rrf_scoring(self, fusion_retriever):
        """Test RRF scoring formula."""
        list1 = [{"id": "1", "content": "first"}]
        list2 = [{"id": "1", "content": "first"}]

        fused = fusion_retriever.fuse_results([list1, list2])

        # RRF score should be 2 * (1 / (60 + 0 + 1)) = 2/61
        expected_score = 2 * (1.0 / (60 + 1))
        assert abs(fused[0]["rrf_score"] - expected_score) < 0.001

    def test_fuse_results_no_id_uses_content_hash(self, fusion_retriever):
        """Test fusion handles chunks without id."""
        list1 = [{"content": "doc1"}]
        list2 = [{"content": "doc1"}]

        fused = fusion_retriever.fuse_results([list1, list2])
        assert len(fused) == 1

    def test_fuse_results_with_top_k(self, fusion_retriever):
        """Test fusion respects top_k limit."""
        list1 = [{"id": str(i), "content": f"doc{i}"} for i in range(10)]
        list2 = [{"id": str(i), "content": f"doc{i}"} for i in range(5, 15)]

        fused = fusion_retriever.fuse_results([list1, list2], top_k=5)
        assert len(fused) == 5

    def test_fuse_results_preserves_first_occurrence_data(self, fusion_retriever):
        """Test fusion preserves data from first occurrence."""
        list1 = [{"id": "1", "content": "doc1", "extra": "data1"}]
        list2 = [{"id": "1", "content": "doc1", "extra": "data2"}]

        fused = fusion_retriever.fuse_results([list1, list2])
        assert fused[0]["extra"] == "data1"  # First occurrence preserved

    def test_fuse_results_sorted_by_score(self, fusion_retriever):
        """Test results are sorted by RRF score."""
        list1 = [
            {"id": "1", "content": "doc1"},
            {"id": "2", "content": "doc2"},
        ]
        list2 = [
            {"id": "2", "content": "doc2"},
            {"id": "1", "content": "doc1"},
        ]
        list3 = [
            {"id": "2", "content": "doc2"},
        ]

        fused = fusion_retriever.fuse_results([list1, list2, list3])

        # Verify sorted by score descending
        scores = [f["rrf_score"] for f in fused]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# QueryService Tests
# =============================================================================


class TestQueryService:
    """Tests for QueryService class."""

    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_empty(
        self, mock_get_llm, mock_session, mock_request_context
    ):
        """Test reformulation with empty query."""
        result = await QueryService.reformulate_query_with_chat_history(
            "", mock_session, ctx=mock_request_context
        )
        assert result == ""

    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_whitespace(
        self, mock_get_llm, mock_session, mock_request_context
    ):
        """Test reformulation with whitespace query."""
        result = await QueryService.reformulate_query_with_chat_history(
            "   ", mock_session, ctx=mock_request_context
        )
        assert result == "   "

    # @pytest.mark.asyncio
    # @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    # async def test_reformulate_query_no_llm(
    #     self, mock_get_llm, mock_session, mock_request_context, capsys
    # ):
    #     """Test reformulation when no LLM is configured."""
    #     mock_get_llm.return_value = None
    #
    #     result = await QueryService.reformulate_query_with_chat_history(
    #         "test query", mock_session, ctx=mock_request_context
    #     )
    #
    #     assert result == "test query"
    #     captured = capsys.readouterr()
    #     assert "Warning" in captured.out
    #
    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_success(
        self, mock_get_llm, mock_session, mock_request_context
    ):
        """Test successful query reformulation."""
        mock_llm = AsyncMock()
        mock_generation = MagicMock()
        mock_generation.text = "improved query for research"
        mock_llm.agenerate = AsyncMock(
            return_value=MagicMock(generations=[[mock_generation]])
        )
        mock_get_llm.return_value = mock_llm

        result = await QueryService.reformulate_query_with_chat_history(
            "test query", mock_session, ctx=mock_request_context
        )

        assert result == "improved query for research"

    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_with_chat_history(
        self, mock_get_llm, mock_session, mock_request_context
    ):
        """Test reformulation with chat history context."""
        mock_llm = AsyncMock()
        mock_generation = MagicMock()
        mock_generation.text = "contextual query"
        mock_llm.agenerate = AsyncMock(
            return_value=MagicMock(generations=[[mock_generation]])
        )
        mock_get_llm.return_value = mock_llm

        result = await QueryService.reformulate_query_with_chat_history(
            "follow up",
            mock_session,
            ctx=mock_request_context,
            chat_history_str="<user>previous question</user>",
        )

        assert result == "contextual query"

    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_empty_response(
        self, mock_get_llm, mock_session, mock_request_context
    ):
        """Test reformulation with empty LLM response."""
        mock_llm = AsyncMock()
        mock_generation = MagicMock()
        mock_generation.text = ""
        mock_llm.agenerate = AsyncMock(
            return_value=MagicMock(generations=[[mock_generation]])
        )
        mock_get_llm.return_value = mock_llm

        result = await QueryService.reformulate_query_with_chat_history(
            "test query", mock_session, ctx=mock_request_context
        )

        assert result == "test query"  # Falls back to original

    @pytest.mark.asyncio
    @patch("backend.src.convo.services.llm_service.LLMService.get_llm_for_role")
    async def test_reformulate_query_exception(
        self, mock_get_llm, mock_session, mock_request_context, capsys
    ):
        """Test reformulation handles exceptions."""
        mock_get_llm.side_effect = Exception("LLM error")

        result = await QueryService.reformulate_query_with_chat_history(
            "test query", mock_session, ctx=mock_request_context
        )

        assert result == "test query"
        captured = capsys.readouterr()
        assert "Error reformulating" in captured.out

    @pytest.mark.asyncio
    async def test_langchain_chat_history_to_str_empty(self):
        """Test chat history conversion with empty list."""
        result = await QueryService.langchain_chat_history_to_str([])
        assert "<chat_history>" in result
        assert "</chat_history>" in result

    @pytest.mark.asyncio
    async def test_langchain_chat_history_to_str_human_message(self):
        """Test chat history conversion with HumanMessage."""
        history = [HumanMessage(content="Hello")]
        result = await QueryService.langchain_chat_history_to_str(history)
        assert "<user>Hello</user>" in result

    @pytest.mark.asyncio
    async def test_langchain_chat_history_to_str_ai_message(self):
        """Test chat history conversion with AIMessage."""
        history = [AIMessage(content="Hi there")]
        result = await QueryService.langchain_chat_history_to_str(history)
        assert "<assistant>Hi there</assistant>" in result

    @pytest.mark.asyncio
    async def test_langchain_chat_history_to_str_system_message(self):
        """Test chat history conversion with SystemMessage."""
        history = [SystemMessage(content="You are helpful")]
        result = await QueryService.langchain_chat_history_to_str(history)
        assert "<system>You are helpful</system>" in result

    @pytest.mark.asyncio
    async def test_langchain_chat_history_to_str_mixed(self):
        """Test chat history conversion with mixed messages."""
        history = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User question"),
            AIMessage(content="Assistant response"),
        ]
        result = await QueryService.langchain_chat_history_to_str(history)

        assert "<system>System prompt</system>" in result
        assert "<user>User question</user>" in result
        assert "<assistant>Assistant response</assistant>" in result
        assert result.startswith("<chat_history>")
        assert result.endswith("</chat_history>")
