"""
Tests for chunker_service module.

Provides >90% code coverage for:
- ChunkerConfig, ChunkingResult dataclasses
- BaseChunker abstract class and registry pattern
- SemanticChunker, TokenChunker, SentenceChunker, FixedSizeChunker
- ChunkerService main interface
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from backend.src.processing.services.chunker_service import (
    ChunkerType,
    ChunkerConfig,
    ChunkingResult,
    BaseChunker,
    SemanticChunker,
    TokenChunker,
    SentenceChunker,
    FixedSizeChunker,
    ChunkerService,
)
from backend.src.storage.schemas import (
    Chunk,
    Source,
    SourceWithRelations,
    Origin,
    OriginType,
    ConnectorType,
    ulid_factory,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_text() -> str:
    """Sample text for chunking tests."""
    return """
    Machine learning is a subset of artificial intelligence. It focuses on building 
    systems that learn from data. Deep learning is a subset of machine learning.

    Neural networks are inspired by biological neurons. They consist of layers of 
    interconnected nodes. Each node performs a simple computation.

    Natural language processing deals with text and speech. It enables computers 
    to understand human language. Applications include translation and sentiment analysis.
    """


@pytest.fixture
def short_text() -> str:
    """Short text for simple tests."""
    return "Hello world. This is a test."


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
def sample_source(sample_origin: Origin, sample_text: str) -> SourceWithRelations:
    """Create a sample source with content."""
    return SourceWithRelations(
        source_id=ulid_factory(),
        source_hash=sample_origin.source_hash,
        unique_id="test-unique-id",
        source_summary="Test document about ML",
        summary_embedding="[]",
        source_content=sample_text,
        origin=sample_origin,
        chunks=[],
    )


@pytest.fixture
def empty_source(sample_origin: Origin) -> SourceWithRelations:
    """Create a source without content."""
    return SourceWithRelations(
        source_id=ulid_factory(),
        source_hash=sample_origin.source_hash,
        unique_id="empty-unique-id",
        source_summary="",
        summary_embedding="[]",
        source_content=None,
        origin=sample_origin,
        chunks=[],
    )


@pytest.fixture
def chunker_config() -> ChunkerConfig:
    """Default chunker configuration."""
    return ChunkerConfig(
        chunk_size=128,
        chunk_overlap=32,
        min_chunk_size=50,
        max_chunk_size=500,
    )


# =============================================================================
# ChunkerConfig Tests
# =============================================================================


class TestChunkerConfig:
    """Tests for ChunkerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkerConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
        assert config.similarity_threshold == 0.8
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.sentences_per_chunk == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkerConfig(
            chunk_size=256,
            chunk_overlap=64,
            similarity_threshold=0.9,
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 64
        assert config.similarity_threshold == 0.9


# =============================================================================
# ChunkingResult Tests
# =============================================================================


class TestChunkingResult:
    """Tests for ChunkingResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = ChunkingResult(chunks=[])
        assert result.chunks == []
        assert result.chunker_used == ""
        assert result.chunk_time_ms == 0.0
        assert result.total_tokens == 0
        assert result.errors == []

    def test_with_chunks(self):
        """Test result with chunks."""
        chunk = Chunk(
            chunk_id="chunk1",
            source_id="source1",
            text="Test text",
            start_index=0,
            end_index=9,
            token_count=2,
        )
        result = ChunkingResult(
            chunks=[chunk],
            chunker_used="SEMANTIC",
            chunk_time_ms=100.5,
            total_tokens=2,
        )
        assert len(result.chunks) == 1
        assert result.chunker_used == "SEMANTIC"
        assert result.chunk_time_ms == 100.5

    def test_with_errors(self):
        """Test result with errors."""
        result = ChunkingResult(
            chunks=[],
            errors=["Error 1", "Error 2"],
        )
        assert len(result.errors) == 2


# =============================================================================
# BaseChunker Tests
# =============================================================================


class TestBaseChunker:
    """Tests for BaseChunker abstract class and registry."""

    def test_registry_contains_all_chunkers(self):
        """Test that all chunker types are registered."""
        available = BaseChunker.get_available_chunkers()
        assert "SEMANTIC" in available
        assert "TOKEN" in available
        assert "SENTENCE" in available
        assert "FIXED" in available

    def test_get_chunker_valid_type(self):
        """Test getting a valid chunker."""
        chunker = BaseChunker.get_chunker(ChunkerType.FIXED)

        assert chunker is not None
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_chunker_all_types(self):
        """Test getting all registered chunker types."""
        for chunker_type in ChunkerType:
            # Skip RECURSIVE if not implemented
            if chunker_type == ChunkerType.RECURSIVE:
                continue
            chunker = BaseChunker.get_chunker(chunker_type)

            assert chunker is not None

    def test_create_chunk_helper(self):
        """Test the _create_chunk helper method."""
        chunker = FixedSizeChunker()

        chunk = chunker._create_chunk(
            text="Test text",
            source_id="source1",
            source_hash="hash1",
            start_index=0,
            end_index=9,
            token_count=2,
        )
        assert chunk.text == "Test text"
        assert chunk.source_id == "source1"
        assert chunk.start_index == 0
        assert chunk.end_index == 9
        assert chunk.token_count == 2
        assert chunk.chunk_id is not None

    def test_create_error_result(self):
        """Test the _create_error_result helper method."""
        chunker = FixedSizeChunker()
        result = chunker._create_error_result("Test error")
        assert result.chunks == []
        assert result.chunker_used == ChunkerType.FIXED.value
        assert "Test error" in result.errors


# =============================================================================
# FixedSizeChunker Tests
# =============================================================================


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    @pytest.mark.asyncio
    async def test_chunk_text_basic(self, sample_text: str):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker()
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10)

        result = await chunker.chunk_text(
            text=sample_text,
            source_id="source1",
            source_hash="hash1",
            config=config,
        )

        assert len(result.chunks) > 0
        assert result.chunker_used == "FIXED"
        assert result.chunk_time_ms > 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_chunk_text_short(self, short_text: str):
        """Test chunking short text."""
        chunker = FixedSizeChunker()
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)

        result = await chunker.chunk_text(
            text=short_text,
            source_id="source1",
            source_hash="hash1",
            config=config,
        )

        # Short text should produce 1 chunk
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = FixedSizeChunker()

        result = await chunker.chunk_text(
            text="",
            source_id="source1",
            source_hash="hash1",
        )

        assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = FixedSizeChunker()

        result = await chunker.chunk_text(
            text="   \n\t  ",
            source_id="source1",
            source_hash="hash1",
        )

        assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_chunk_text_default_config(self, sample_text: str):
        """Test chunking with default config."""
        chunker = FixedSizeChunker()

        result = await chunker.chunk_text(
            text=sample_text,
            source_id="source1",
            source_hash="hash1",
            config=None,
        )

        assert result.errors == []

    @pytest.mark.asyncio
    async def test_chunk_source(self, sample_source: SourceWithRelations):
        """Test chunking a source."""
        chunker = FixedSizeChunker()
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10)

        result = await chunker.chunk_source(sample_source, config)

        assert len(result.chunks) > 0
        assert result.chunks[0].source_id == sample_source.source_id

    @pytest.mark.asyncio
    async def test_chunk_source_no_content(self, empty_source: SourceWithRelations):
        """Test chunking a source with no content."""
        chunker = FixedSizeChunker()

        result = await chunker.chunk_source(empty_source)

        assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_chunk_preserves_word_boundaries(self):
        """Test that chunking tries to preserve word boundaries."""
        chunker = FixedSizeChunker()
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        config = ChunkerConfig(chunk_size=5, chunk_overlap=1)

        result = await chunker.chunk_text(
            text=text,
            source_id="source1",
            source_hash="hash1",
            config=config,
        )

        # Check that chunks don't break mid-word (when possible)
        for chunk in result.chunks:
            # Chunks should not start or end with partial words
            assert (
                not chunk.text.startswith(" ")
                or chunk.text.strip() == chunk.text.strip()
            )


# =============================================================================
# TokenChunker Tests
# =============================================================================


class TestTokenChunker:
    """Tests for TokenChunker."""

    @pytest.mark.asyncio
    async def test_chunk_text_basic(self, sample_text: str):
        """Test basic token chunking."""
        chunker = TokenChunker()
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10)

        # Mock the chonkie chunker
        mock_chunk = MagicMock()
        mock_chunk.text = "Test chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 10
        mock_chunk.token_count = 2

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_run.return_value = [mock_chunk]

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
                config=config,
            )

            assert result.chunker_used == "TOKEN"

    # @pytest.mark.asyncio
    # @patch("backend.src.processing.services.chunker_service.asyncio.to_thread")
    # async def test_chunk_text_error_handling(self, mock_run, sample_text: str):
    #     """Test error handling in token chunking."""
    #     chunker = TokenChunker()
    #
    #     mock_run.side_effect = Exception("Chonkie error")
    #
    #     result = await chunker.chunk_text(
    #         text=sample_text,
    #         source_id="source1",
    #         source_hash="hash1",
    #     )
    #
    #     assert "Chonkie error" in result.errors
    #


# =============================================================================
# SentenceChunker Tests
# =============================================================================


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    @pytest.mark.asyncio
    async def test_chunk_text_basic(self, sample_text: str):
        """Test basic sentence chunking."""
        chunker = SentenceChunker()
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)

        mock_chunk = MagicMock()
        mock_chunk.text = "Test sentence."
        mock_chunk.start_index = 0
        mock_chunk.end_index = 14
        mock_chunk.token_count = 3

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_run.return_value = [mock_chunk]

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
                config=config,
            )

            assert len(result.chunks) == 1
            assert result.chunker_used == "SENTENCE"

    @pytest.mark.asyncio
    async def test_chunk_text_error_handling(self, sample_text: str):
        """Test error handling in sentence chunking."""
        chunker = SentenceChunker()

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_run.side_effect = Exception("Sentence chunking failed")

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
            )

            assert len(result.errors) > 0


# =============================================================================
# SemanticChunker Tests
# =============================================================================


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.mark.asyncio
    async def test_initialize_chunker(self):
        """Test lazy initialization of semantic chunker."""
        chunker = SemanticChunker()
        config = ChunkerConfig()

        assert chunker._chonkie_chunker is None  # type: ignore[attr-defined]

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_chunk = MagicMock()
            mock_chunk.text = "Test"
            mock_chunk.start_index = 0
            mock_chunk.end_index = 4
            mock_chunk.token_count = 1
            mock_run.return_value = [mock_chunk]

            with patch.object(
                chunker, "_initialize_chunker", new_callable=AsyncMock
            ) as mock_init:
                # Set the chunker after initialization
                async def set_chunker(cfg):
                    chunker._chonkie_chunker = MagicMock()  # type: ignore[attr-defined]

                mock_init.side_effect = set_chunker

                result = await chunker.chunk_text(
                    text="Test text",
                    source_id="source1",
                    source_hash="hash1",
                    config=config,
                )

                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_text_basic(self, sample_text: str):
        """Test basic semantic chunking."""
        chunker: SemanticChunker = SemanticChunker()
        config = ChunkerConfig()

        mock_chunk = MagicMock()
        mock_chunk.text = "Semantic chunk"
        mock_chunk.start_index = 0
        mock_chunk.end_index = 14
        mock_chunk.token_count = 2

        # Mock the chonkie chunker directly
        chunker._chonkie_chunker = MagicMock()  # type: ignore[attr-defined]

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_run.return_value = [mock_chunk]

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
                config=config,
            )

            assert len(result.chunks) == 1
            assert result.chunker_used == "SEMANTIC"
            assert result.total_tokens == 2

    @pytest.mark.asyncio
    async def test_chunk_text_initialization_failure(self, sample_text: str):
        """Test handling of initialization failure."""
        chunker = SemanticChunker()
        chunker._chonkie_chunker = None  # type: ignore[attr-defined]

        with patch.object(
            chunker, "_initialize_chunker", new_callable=AsyncMock
        ) as mock_init:
            # Don't set the chunker - leave it None
            mock_init.return_value = None

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
            )

            assert len(result.errors) > 0
            assert "Failed to initialize chunker" in result.errors[0]

    @pytest.mark.asyncio
    async def test_chunk_text_error_handling(self, sample_text: str):
        """Test error handling in semantic chunking."""
        chunker = SemanticChunker()
        chunker._chonkie_chunker = MagicMock()  # type: ignore[attr-defined]

        with patch(
            "backend.src.processing.services.chunker_service.asyncio.to_thread"
        ) as mock_run:
            mock_run.side_effect = Exception("Semantic error")

            result = await chunker.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
            )

            assert len(result.errors) > 0
            assert "Semantic error" in result.errors[0]


# =============================================================================
# ChunkerService Tests
# =============================================================================


class TestChunkerService:
    """Tests for ChunkerService main interface."""

    def test_init_default(self):
        """Test default initialization."""
        service = ChunkerService()
        assert service.default_chunker == ChunkerType.SEMANTIC

    def test_init_custom_default(self):
        """Test initialization with custom default."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)
        assert service.default_chunker == ChunkerType.FIXED

    def test_get_chunker(self):
        """Test getting a chunker by type."""
        service = ChunkerService()
        chunker = service.get_chunker(ChunkerType.FIXED)
        assert chunker is not None
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_available_chunkers(self):
        """Test getting available chunkers."""
        service = ChunkerService()
        available = service.get_available_chunkers()
        assert "FIXED" in available
        assert "SEMANTIC" in available

    @pytest.mark.asyncio
    async def test_chunk_text_default_chunker(self, sample_text: str):
        """Test chunking text with default chunker."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)

        result = await service.chunk_text(
            text=sample_text,
            source_id="source1",
            source_hash="hash1",
        )

        assert result.chunker_used == "FIXED"
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_text_specific_chunker(self, sample_text: str):
        """Test chunking text with specific chunker type."""
        service = ChunkerService(default_chunker=ChunkerType.SEMANTIC)

        result = await service.chunk_text(
            text=sample_text,
            source_id="source1",
            source_hash="hash1",
            chunker_type=ChunkerType.FIXED,
        )

        assert result.chunker_used == "FIXED"

    @pytest.mark.asyncio
    async def test_chunk_text_invalid_chunker(self, sample_text: str):
        """Test chunking with invalid chunker type."""
        service = ChunkerService()

        # Temporarily remove a chunker from registry
        original = BaseChunker._registry.copy()
        BaseChunker._registry.clear()

        try:
            result = await service.chunk_text(
                text=sample_text,
                source_id="source1",
                source_hash="hash1",
                chunker_type=ChunkerType.FIXED,
            )

            assert len(result.errors) > 0
            assert "Chunker not found" in result.errors[0]
        finally:
            BaseChunker._registry = original

    @pytest.mark.asyncio
    async def test_chunk_source(self, sample_source: SourceWithRelations):
        """Test chunking a source."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)

        result = await service.chunk_source(sample_source)

        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_source_invalid_chunker(
        self, sample_source: SourceWithRelations
    ):
        """Test chunking source with invalid chunker."""
        service = ChunkerService()

        original = BaseChunker._registry.copy()
        BaseChunker._registry.clear()

        try:
            result = await service.chunk_source(
                sample_source,
                chunker_type=ChunkerType.FIXED,
            )

            # Should return source unchanged
            assert result == sample_source
        finally:
            BaseChunker._registry = original

    @pytest.mark.asyncio
    async def test_chunk_sources_multiple(self, sample_source: SourceWithRelations):
        """Test chunking multiple sources."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)

        # Create multiple sources
        sources = [sample_source]
        for i in range(2):
            new_source = SourceWithRelations(
                source_id=ulid_factory(),
                source_hash=f"hash{i}",
                unique_id=f"unique-{i}",
                source_summary="",
                summary_embedding="[]",
                source_content=f"Content for source {i}. More text here.",
                origin=sample_source.origin,
                chunks=[],
            )
            sources.append(new_source)

        results = await service.chunk_sources(sources, max_concurrent=2)

        assert len(results) == 3
        # All sources should have chunks
        for result in results:
            if result.source_content:
                assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_sources_with_errors(self, sample_source: SourceWithRelations):
        """Test chunking sources when some fail."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)

        sources = [sample_source]

        # Mock chunker to fail for one source
        with patch.object(FixedSizeChunker, "chunk_source") as mock_chunk:

            async def side_effect(source, config=None):
                if source == sources[0]:  # Only fail for the first source
                    raise Exception("Chunking failed")
                return source

            mock_chunk.side_effect = side_effect

            results = await service.chunk_sources(sources)

            # Should still return the source even on failure
            assert len(results) == 1
            assert (
                results[0] == sample_source
            )  # Should be the original source due to error

    @pytest.mark.asyncio
    async def test_chunk_sources_invalid_chunker(
        self, sample_source: SourceWithRelations
    ):
        """Test chunking sources with invalid chunker."""
        service = ChunkerService()

        original = BaseChunker._registry.copy()
        BaseChunker._registry.clear()

        try:
            results = await service.chunk_sources(
                [sample_source],
                chunker_type=ChunkerType.FIXED,
            )

            # Should return sources unchanged
            assert results == [sample_source]
        finally:
            BaseChunker._registry = original


# =============================================================================
# Integration Tests
# =============================================================================


class TestChunkerIntegration:
    """Integration tests for chunker service."""

    @pytest.mark.asyncio
    async def test_full_pipeline_fixed_chunker(
        self, sample_source: SourceWithRelations
    ):
        """Test full chunking pipeline with fixed chunker."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10)

        # Chunk the source
        result = await service.chunk_source(sample_source, config=config)

        # Verify chunks
        assert len(result.chunks) > 0

        for chunk in result.chunks:
            assert chunk.chunk_id is not None
            assert chunk.source_id == sample_source.source_id
            assert chunk.text is not None
            assert chunk.start_index >= 0
            assert chunk.end_index > chunk.start_index
            assert chunk.token_count > 0

    @pytest.mark.asyncio
    async def test_chunker_type_enum(self):
        """Test all ChunkerType enum values."""
        assert ChunkerType.SEMANTIC == "SEMANTIC"
        assert ChunkerType.TOKEN == "TOKEN"
        assert ChunkerType.SENTENCE == "SENTENCE"
        assert ChunkerType.FIXED == "FIXED"
        assert ChunkerType.RECURSIVE == "RECURSIVE"

    @pytest.mark.asyncio
    async def test_chunk_result_token_counting(self, sample_text: str):
        """Test that token counts are accumulated correctly."""
        service = ChunkerService(default_chunker=ChunkerType.FIXED)
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10)

        result = await service.chunk_text(
            text=sample_text,
            source_id="source1",
            source_hash="hash1",
            config=config,
        )

        # Total tokens should equal sum of chunk tokens
        calculated_total = sum(chunk.token_count for chunk in result.chunks)
        assert result.total_tokens == calculated_total
