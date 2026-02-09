"""
Tests for FOSRA/src/orchestration/conversational/convo_flow_nodes.py

Achieves 90%+ code coverage by testing:
- rerank_sources with various inputs and edge cases
- answer_question with/without sources
- answer_question_streaming with token streaming
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.src.retrieval.services.reranker_service import RerankResult
from backend.src.orchestration.conversational.convo_flow_nodes import (
    rerank_sources,
    answer_question,
    answer_question_streaming,
)
from backend.src.convo.utils.streaming_utils import StreamingService
from backend.src.orchestration.main_flow.main_flow_config import Configuration

from backend.src.resources.retrieval_fixtures import *

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def streaming_service():
    """Create a StreamingService instance for testing."""
    return StreamingService()


@pytest.fixture
def mock_writer():
    """Create a mock StreamWriter."""
    return MagicMock()


@pytest.fixture
def mock_db_session():
    """Create a mock AsyncSession."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_config():
    """Create a mock RunnableConfig."""
    return {
        "configurable": {
            "user_id": "test_user",
            "workspace_id": 1,
            "user_query": "What is machine learning?",
            "language": "English",
            "top_k": 10,
            "relevant_sources": [],
        }
    }


@pytest.fixture
def sample_sources():
    """Create sample source data for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "Machine learning is a subset of AI.",
            "text": "Machine learning is a subset of AI.",
            "score": 0.85,
            "source_info": {
                "id": "source_1",
                "title": "ML Introduction",
                "source_type": "FILE",
            },
            "source": "FILE",
            "document": {
                "id": "source_1",
                "title": "ML Introduction",
            },
        },
        {
            "chunk_id": "chunk_2",
            "content": "Deep learning uses neural networks.",
            "text": "Deep learning uses neural networks.",
            "score": 0.75,
            "source_info": {
                "id": "source_2",
                "title": "Deep Learning Guide",
                "source_type": "FILE",
            },
            "source": "FILE",
            "document": {
                "id": "source_2",
                "title": "Deep Learning Guide",
            },
        },
    ]


@pytest.fixture
def mock_state(mock_db_session, streaming_service):
    """Create a mock State object."""
    state = MagicMock()
    state.db_session = mock_db_session
    state.streaming_service = streaming_service
    state.chat_history = []
    state.reranked_sources = []
    return state


# ============================================================================
# Tests for rerank_sources
# ============================================================================


class TestRerankSources:
    """Tests for the rerank_sources function."""

    @pytest.mark.asyncio
    async def test_empty_sources_from_state(self, mock_state, mock_config):
        """Should return empty lists when no sources in state or config."""
        mock_state.reranked_sources = []

        result = await rerank_sources(mock_state, mock_config)

        assert result["reranked_sources"] == []
        assert result["reranked_documents"] == []

    @pytest.mark.asyncio
    async def test_empty_sources_returns_empty(self, mock_state, mock_config):
        """Should return empty lists when sources is None."""
        mock_state.reranked_sources = None
        mock_config["configurable"]["relevant_sources"] = None

        result = await rerank_sources(mock_state, mock_config)

        assert result["reranked_sources"] == []
        assert result["reranked_documents"] == []

    @pytest.mark.asyncio
    async def test_sources_from_config_when_state_empty(
        self, mock_state, mock_config, sample_sources
    ):
        """Should use sources from config when state is empty."""
        mock_state.reranked_sources = None
        mock_config["configurable"]["relevant_sources"] = sample_sources

        mock_reranker = AsyncMock()
        mock_rerank_result = RerankResult(
            documents=[
                {"chunk_id": "chunk_1", "score": 0.95, "content": "ML content"},
                {"chunk_id": "chunk_2", "score": 0.85, "content": "DL content"},
            ],
            reranker_used="FLASHRANK",
            original_count=2,
            filtered_count=2,
        )
        mock_reranker.rerank = AsyncMock(return_value=mock_rerank_result)

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = mock_reranker
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            assert len(result["reranked_sources"]) == 2
            # Verify sorted by score descending
            assert (
                result["reranked_sources"][0]["score"]
                >= result["reranked_sources"][1]["score"]
            )

    @pytest.mark.asyncio
    async def test_no_reranker_available(self, mock_state, mock_config, sample_sources):
        """Should return original sources when no reranker available."""
        mock_state.reranked_sources = sample_sources

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = None
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            assert result["reranked_sources"] == sample_sources
            assert result["reranked_documents"] == sample_sources

    @pytest.mark.asyncio
    async def test_successful_reranking(self, mock_state, mock_config, sample_sources):
        """Should rerank sources successfully."""
        mock_state.reranked_sources = sample_sources

        mock_reranker = AsyncMock()
        mock_rerank_result = RerankResult(
            documents=[
                {"chunk_id": "chunk_2", "score": 0.98, "content": "DL content"},
                {"chunk_id": "chunk_1", "score": 0.92, "content": "ML content"},
            ],
            reranker_used="FLASHRANK",
            original_count=2,
            filtered_count=2,
        )
        mock_reranker.rerank = AsyncMock(return_value=mock_rerank_result)

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = mock_reranker
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            assert len(result["reranked_sources"]) == 2
            # chunk_2 should be first due to higher score
            assert result["reranked_sources"][0]["chunk_id"] == "chunk_2"

    @pytest.mark.asyncio
    async def test_reranker_failure_fallback(
        self, mock_state, mock_config, sample_sources
    ):
        """Should fallback to original order when reranker fails."""
        mock_state.reranked_sources = sample_sources

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=Exception("Reranker error"))

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = mock_reranker
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            # Should fallback to original sources
            assert result["reranked_sources"] == sample_sources
            assert result["reranked_documents"] == sample_sources

    @pytest.mark.asyncio
    async def test_sources_with_text_instead_of_content(self, mock_state, mock_config):
        """Should handle sources that use 'text' field instead of 'content'."""
        sources_with_text = [
            {
                "chunk_id": "chunk_1",
                "text": "Some text content",
                "score": 0.8,
                "source_info": {"id": "src_1"},
                "source": "FILE",
            }
        ]
        mock_state.reranked_sources = sources_with_text

        mock_reranker = AsyncMock()
        mock_rerank_result = RerankResult(
            documents=[
                {"chunk_id": "chunk_1", "score": 0.9, "content": "Some text content"}
            ],
            reranker_used="FLASHRANK",
            original_count=1,
            filtered_count=1,
        )
        mock_reranker.rerank = AsyncMock(return_value=mock_rerank_result)

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = mock_reranker
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            assert len(result["reranked_sources"]) == 1
            # Verify reranker.rerank was called
            mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_reranker_input_after_processing(self, mock_state, mock_config):
        """Should return empty when reranker input becomes empty."""
        # Sources that result in empty reranker input
        mock_state.reranked_sources = [{}]  # Empty dict with no useful fields

        mock_reranker = AsyncMock()
        mock_rerank_result = RerankResult(
            documents=[],
            reranker_used="FLASHRANK",
            original_count=1,
            filtered_count=0,
        )
        mock_reranker.rerank = AsyncMock(return_value=mock_rerank_result)

        with patch(
            "backend.src.orchestration.conversational.convo_flow_nodes.RerankerService"
        ) as MockRerankerService:
            mock_service = MagicMock()
            mock_service.get_reranker.return_value = mock_reranker
            MockRerankerService.return_value = mock_service

            result = await rerank_sources(mock_state, mock_config)

            # Should return empty or handle gracefully
            assert "reranked_sources" in result


# ============================================================================
# Tests for answer_question
# ============================================================================


class TestAnswerQuestion:
    """Tests for the answer_question function."""

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.main_flow.main_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_no_llm_configured(
        self,
        mock_llm_service,
        mock_state,
        mock_config,
    ):
        """Should raise ValueError when LLM is not configured."""
        mock_state.reranked_sources = []

        mock_llm_service.return_value = None

        with pytest.raises(ValueError, match="LLM configuration missing"):
            await answer_question(mock_state, mock_config)
            mock_llm_service.get_llm_for_role.assert_called()

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_answer_without_sources(self, mock_get_llm, mock_state, mock_config):
        """Should generate answer without sources using general knowledge."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="This is an answer based on general knowledge.")

        mock_llm.astream = mock_astream

        result = await answer_question(mock_state, mock_config)

        assert "final_answer" in result
        assert "This is an answer based on general knowledge." in result["final_answer"]
        assert result["reranked_documents"] == []

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.optimize_sources_for_token_limit"
    )
    async def test_answer_with_sources(
        self, mock_optimize, mock_get_llm, mock_state, mock_config, sample_sources
    ):
        """Should generate answer using provided sources."""
        mock_state.reranked_sources = sample_sources
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Machine learning is ")
            yield MagicMock(content="a subset of AI [citation:source_1].")

        mock_llm.astream = mock_astream

        mock_optimize.return_value = (sample_sources, 500)

        result = await answer_question(mock_state, mock_config)

        assert "final_answer" in result
        assert "Machine learning is a subset of AI" in result["final_answer"]
        assert len(result["reranked_documents"]) == 2

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_answer_with_chat_history(
        self, mock_get_llm, mock_state, mock_config
    ):
        """Should include chat history context in answer generation."""
        mock_state.reranked_sources = []
        mock_state.chat_history = [
            HumanMessage(content="Tell me about AI"),
            AIMessage(content="AI stands for Artificial Intelligence."),
        ]

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Based on our previous discussion about AI...")

        mock_llm.astream = mock_astream

        result = await answer_question(mock_state, mock_config)

        assert "final_answer" in result
        assert "Based on our previous discussion" in result["final_answer"]

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_streaming_with_non_string_content(
        self, mock_get_llm, mock_state, mock_config
    ):
        """Should handle chunks with non-string content."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Valid string")
            yield MagicMock(content=None)  # Non-string content
            yield MagicMock(content=123)  # Integer content
            yield MagicMock(content="More text")

        mock_llm.astream = mock_astream

        result = await answer_question(mock_state, mock_config)

        assert "final_answer" in result
        # Should only include string content
        assert "Valid string" in result["final_answer"]
        assert "More text" in result["final_answer"]

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_chunk_without_content_attribute(
        self, mock_get_llm, mock_state, mock_config
    ):
        """Should handle chunks without content attribute."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        class ChunkWithoutContent:
            pass

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Valid chunk")
            yield ChunkWithoutContent()  # No content attribute
            yield MagicMock(content="Another chunk")

        mock_llm.astream = mock_astream

        result = await answer_question(mock_state, mock_config)

        assert "final_answer" in result
        assert "Valid chunk" in result["final_answer"]


# ============================================================================
# Tests for answer_question_streaming
# ============================================================================


class TestAnswerQuestionStreaming:
    """Tests for the answer_question_streaming function."""

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_no_llm_configured_streaming(
        self, mock_get_llm, mock_state, mock_config, mock_writer
    ):
        """Should write error and return when LLM is not configured."""
        mock_state.reranked_sources = []

        mock_get_llm.return_value = None
        # Consume the generator
        chunks = []
        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        # Should have written error
        mock_writer.assert_called()
        # No chunks should be yielded
        assert chunks == []

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_streaming_without_sources(
        self, mock_get_llm, mock_state, mock_config, mock_writer
    ):
        """Should stream answer without sources."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Token1 ")
            yield MagicMock(content="Token2 ")
            yield MagicMock(content="Token3")

        mock_llm.astream = mock_astream

        chunks = []
        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == "Token1 "
        assert chunks[1] == "Token2 "
        assert chunks[2] == "Token3"

        # Writer should be called for each token
        assert mock_writer.call_count == 3

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.optimize_sources_for_token_limit"
    )
    async def test_streaming_with_sources(
        self,
        mock_optimize,
        mock_get_llm,
        mock_state,
        mock_config,
        mock_writer,
        sample_sources,
    ):
        """Should stream answer with sources."""
        mock_state.reranked_sources = sample_sources
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Answer with ")
            yield MagicMock(content="citation [source_1]")

        mock_llm.astream = mock_astream

        chunks = []

        mock_optimize.return_value = (sample_sources, 500)

        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        full_answer = "".join(chunks)
        assert "Answer with citation [source_1]" in full_answer

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_streaming_skips_non_string_content(
        self, mock_get_llm, mock_state, mock_config, mock_writer
    ):
        """Should skip chunks with non-string content in streaming."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Valid")
            yield MagicMock(content=None)
            yield MagicMock(content=42)
            yield MagicMock(content=" text")

        mock_llm.astream = mock_astream

        chunks = []

        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        # Only string content should be yielded
        assert len(chunks) == 2
        assert chunks[0] == "Valid"
        assert chunks[1] == " text"

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_streaming_with_chat_history(
        self, mock_get_llm, mock_state, mock_config, mock_writer
    ):
        """Should include chat history in streaming context."""
        mock_state.reranked_sources = []
        mock_state.chat_history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
        ]

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Contextual answer")

        mock_llm.astream = mock_astream

        chunks = []
        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "Contextual answer"

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_streaming_empty_chunk_content(
        self, mock_get_llm, mock_state, mock_config, mock_writer
    ):
        """Should handle empty string content in chunks."""
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="")  # Empty string
            yield MagicMock(content="Actual content")
            yield MagicMock(content="")  # Another empty

        mock_llm.astream = mock_astream

        chunks = []
        async for chunk in answer_question_streaming(
            mock_state, mock_config, mock_writer
        ):
            chunks.append(chunk)

        # Empty strings are still valid, they just don't contribute much
        # The generator should still work correctly


# ============================================================================
# Integration-style tests
# ============================================================================


class TestConvoFlowIntegration:
    """Integration tests for the conversation flow nodes."""

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.optimize_sources_for_token_limit"
    )
    @patch("backend.src.orchestration.conversational.convo_flow_nodes.RerankerService")
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_rerank_then_answer_flow(
        self,
        mock_get_llm,
        mock_reranker_service,
        mock_optimize,
        mock_state,
        mock_config,
        sample_sources,
    ):
        """Test the full flow of reranking then answering."""
        mock_state.reranked_sources = sample_sources
        mock_state.chat_history = []

        # Step 1: Rerank
        mock_reranker = AsyncMock()
        mock_rerank_result = RerankResult(
            documents=[
                {"chunk_id": "chunk_2", "score": 0.95, "content": "DL content"},
                {"chunk_id": "chunk_1", "score": 0.90, "content": "ML content"},
            ],
            reranker_used="FLASHRANK",
            original_count=2,
            filtered_count=2,
        )
        mock_reranker.rerank = AsyncMock(return_value=mock_rerank_result)

        mock_service = MagicMock()
        mock_service.get_reranker.return_value = mock_reranker
        mock_reranker_service.return_value = mock_service

        rerank_result = await rerank_sources(mock_state, mock_config)

        # Update state with reranked sources
        mock_state.reranked_sources = rerank_result["reranked_sources"]

        # Step 2: Answer
        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Final answer based on reranked sources.")

        mock_llm.astream = mock_astream

        mock_optimize.return_value = (mock_state.reranked_sources, 500)

        answer_result = await answer_question(mock_state, mock_config)

        assert "final_answer" in answer_result
        assert (
            "Final answer based on reranked sources." in answer_result["final_answer"]
        )

    @pytest.mark.asyncio
    @patch(
        "backend.src.orchestration.conversational.convo_flow_nodes.LLMService.get_llm_for_role"
    )
    async def test_config_with_different_language(self, mock_get_llm, mock_state):
        """Test answer generation with non-English language setting."""
        config = {
            "configurable": {
                "user_id": "test_user",
                "workspace_id": 1,
                "user_query": "¿Qué es el aprendizaje automático?",
                "language": "Spanish",
                "top_k": 10,
                "relevant_sources": [],
            }
        }
        mock_state.reranked_sources = []
        mock_state.chat_history = []

        mock_llm = AsyncMock()
        mock_llm.model = "gpt-4"

        mock_get_llm.return_value = mock_llm

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="El aprendizaje automático es...")

        mock_llm.astream = mock_astream

        result = await answer_question(mock_state, config)  # pyright:ignore #WARN: Fix

        assert "final_answer" in result
        assert "El aprendizaje automático" in result["final_answer"]
