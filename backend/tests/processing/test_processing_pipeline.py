"""
Tests for processing_pipeline.py - Unified document processing orchestration.

Covers:
- Configuration dataclasses (ProcessingConfig, ProcessingResult)
- ChunkingStrategy enum
- ProcessingPipeline initialization and document processing
- Batch processing with concurrency control
- Private helper methods
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest

from backend.src.processing.services.processing_pipeline import (
    ChunkingStrategy,
    ProcessingConfig,
    ProcessingPipeline,
    ProcessingResult,
    RequestContext,
)
from backend.src.processing.services.parser_service import (
    ParsedDocument,
    ParserConfig,
    ParserType,
)
from backend.src.processing.services.embedder_service import (
    EmbedderConfig,
    EmbedderType,
)
from backend.src.storage.schemas import (
    Chunk,
    Origin,
    OriginType,
    OriginType,
    SourceWithRelations,
)


from backend.src.resources.test_fixtures import *


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock database session."""
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_vector_service() -> AsyncMock:
    """Mock vector repository."""
    repo = AsyncMock()
    repo.store_vectors = AsyncMock(return_value=None)
    repo.upsert = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def sample_markdown_path(tmp_path: Path) -> str:
    """Create a temporary markdown file with content."""
    md_path = tmp_path / "test_document.md"
    md_path.write_text("# Test Document\n\nThis is test content for processing.")
    return str(md_path)


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Create a temporary PDF path."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.touch()
    return str(pdf_path)


@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Default processing configuration."""
    return ProcessingConfig(
        parser_type=ParserType.MARKDOWN,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=512,
        chunk_overlap=128,
        embedder_type=EmbedderType.FASTEMBED,
        generate_summary=False,
        store_vectors=False,
    )


@pytest.fixture
def parsed_document() -> ParsedDocument:
    """Sample parsed document."""
    return ParsedDocument(
        content="# Test Document\n\nThis is test content for processing.",
        metadata={"author": "Test"},
        page_count=1,
        parser_used="MARKDOWN",
        parse_time_ms=50.0,
        tables=[],
        images=[],
        errors=[],
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    """Sample chunk for testing."""
    return Chunk(
        chunk_id="chunk_001",
        source_id="source_001",
        source_hash="hash123",
        text="This is test content for processing.",
        start_index=0,
        end_index=40,
        token_count=8,
        dense_vector=[0.1] * 384,
    )


@pytest.fixture
def sample_source(sample_chunk: Chunk) -> SourceWithRelations:
    """Sample source with chunks."""
    return SourceWithRelations(
        source_id="source_001",
        source_hash="hash123",
        unique_id="unique_001",
        source_content="# Test Document\n\nThis is test content.",
        source_summary="",
        summary_embedding="",
        workspace_ids=[1],
        chunks=[sample_chunk],
        origin=Origin(
            name="test.md",
            origin_path="/path/to/test.md",
            origin_type=OriginType.FILE,
            source_hash="hash123",
        ),
    )


# ============================================================================
# Enum Tests
# ============================================================================


class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_chunking_strategies(self):
        """Test all chunking strategy values."""
        assert ChunkingStrategy.SEMANTIC == "SEMANTIC"
        assert ChunkingStrategy.TOKEN == "TOKEN"
        assert ChunkingStrategy.SENTENCE == "SENTENCE"
        assert ChunkingStrategy.FIXED == "FIXED"

    def test_enum_iteration(self):
        """Test that all strategies are iterable."""
        strategies = list(ChunkingStrategy)
        assert len(strategies) == 4


# ============================================================================
# ProcessingConfig Tests
# ============================================================================


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()

        assert config.parser_type is None
        assert config.parser_config is None
        assert config.chunking_strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.embedder_config is None
        assert config.generate_summary is True
        assert config.store_vectors is True

    def test_custom_values(self):
        """Test custom configuration values."""
        parser_config = ParserConfig(max_pages=10)
        embedder_config = EmbedderConfig(batch_size=64)

        config = ProcessingConfig(
            parser_type=ParserType.DOCLING,
            parser_config=parser_config,
            chunking_strategy=ChunkingStrategy.TOKEN,
            chunk_size=256,
            chunk_overlap=64,
            embedder_type=EmbedderType.OPENAI,
            embedder_config=embedder_config,
            generate_summary=False,
            store_vectors=False,
        )

        assert config.parser_type == ParserType.DOCLING
        assert config.parser_config == parser_config
        assert config.chunking_strategy == ChunkingStrategy.TOKEN
        assert config.chunk_size == 256
        assert config.chunk_overlap == 64
        assert config.embedder_type == EmbedderType.OPENAI
        assert config.embedder_config == embedder_config
        assert config.generate_summary is False
        assert config.store_vectors is False


# ============================================================================
# ProcessingResult Tests
# ============================================================================


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = ProcessingResult()

        assert result.source is None
        assert result.parsed_document is None
        assert result.chunk_count == 0
        assert result.page_count == 0
        assert result.processing_time_ms == 0.0
        assert result.errors == []
        assert result.warnings == []

    def test_success_property_true(self, sample_source: SourceWithRelations):
        """Test success property when processing succeeded."""
        result = ProcessingResult(
            source=sample_source,
            chunk_count=1,
            errors=[],
        )

        assert result.success is True

    def test_success_property_false_no_source(self):
        """Test success property when source is None."""
        result = ProcessingResult(
            source=None,
            errors=[],
        )

        assert result.success is False

    def test_success_property_false_with_errors(
        self, sample_source: SourceWithRelations
    ):
        """Test success property when there are errors."""
        result = ProcessingResult(
            source=sample_source,
            errors=["Something went wrong"],
        )

        assert result.success is False

    def test_full_result(
        self, sample_source: SourceWithRelations, parsed_document: ParsedDocument
    ):
        """Test fully populated result."""
        result = ProcessingResult(
            source=sample_source,
            parsed_document=parsed_document,
            chunk_count=5,
            page_count=3,
            processing_time_ms=1500.5,
            errors=[],
            warnings=["Minor warning"],
        )

        assert result.source == sample_source
        assert result.parsed_document == parsed_document
        assert result.chunk_count == 5
        assert result.page_count == 3
        assert result.processing_time_ms == 1500.5
        assert len(result.warnings) == 1
        assert result.success is True


# ============================================================================
# ProcessingPipeline Initialization Tests
# ============================================================================


class TestProcessingPipelineInit:
    """Tests for ProcessingPipeline initialization."""

    def test_init_with_session(self, mock_session: AsyncMock):
        """Test initialization with database session."""
        pipeline = ProcessingPipeline(mock_session)

        assert pipeline.session == mock_session
        assert pipeline.vector_service is None
        assert pipeline.parser_service is not None
        assert pipeline.embedder_service is not None

    def test_init_with_vector_service(
        self, mock_session: AsyncMock, mock_vector_service: AsyncMock
    ):
        """Test initialization with vector repository."""
        pipeline = ProcessingPipeline(mock_session, mock_vector_service)

        assert pipeline.session == mock_session
        assert pipeline.vector_service == mock_vector_service


# ============================================================================
# ProcessingPipeline.process_document Tests
# ============================================================================


class TestProcessDocumentSuccess:
    """Tests for successful document processing."""

    @pytest.mark.asyncio
    async def test_process_document_basic(
        self,
        mock_session: AsyncMock,
        sample_markdown_path: str,
        mock_request_context,
    ):
        """Test basic document processing flow."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        # Mock the parser service
        mock_parsed = ParsedDocument(
            content="Test content",
            page_count=1,
            parser_used="MARKDOWN",
            errors=[],
        )

        # Mock the chunker
        mock_chunk = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test content",
            start_index=0,
            end_index=12,
            token_count=2,
            dense_vector=[0.1] * 384,
        )

        mock_source = SourceWithRelations(
            source_id="source_001",
            source_hash="hash123",
            unique_id="unique_001",
            source_content="Test content",
            source_summary="",
            summary_embedding="",
            workspace_ids=[ctx.workspace_id],
            chunks=[mock_chunk],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker_instance = MagicMock()
                mock_chunker_instance.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker_instance

                with patch.object(
                    pipeline.embedder_service,
                    "get_embedder",
                    return_value=MagicMock(
                        embed_source=AsyncMock(return_value=mock_source)
                    ),
                ):
                    result = await pipeline.process_document(
                        file_path=sample_markdown_path, name="test.md", ctx=ctx
                    )

        assert result.success is True
        assert result.source is not None
        assert result.chunk_count == 1
        assert result.page_count == 1
        assert result.processing_time_ms > 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_process_document_with_summary(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test document processing with summary generation."""
        ctx = mock_request_context
        ctx.preferences.parser.generate_summary = True

        pipeline = ProcessingPipeline(mock_session)

        mock_parsed = ParsedDocument(
            content="Test content for summary",
            page_count=1,
            parser_used="MARKDOWN",
            errors=[],
        )

        mock_chunk = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test content",
            start_index=0,
            end_index=12,
            token_count=2,
            dense_vector=[0.1] * 384,
        )

        mock_source = SourceWithRelations(
            source_id="source_001",
            source_hash="hash123",
            unique_id="unique_001",
            source_content="Test content for summary",
            source_summary="",
            summary_embedding="",
            workspace_ids=[ctx.workspace_id],
            chunks=[mock_chunk],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker_instance = MagicMock()
                mock_chunker_instance.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker_instance

                with patch(
                    "backend.src.convo.services.llm_service.LLMService.get_llm_for_role",
                    return_value=MagicMock(),
                ):
                    with patch(
                        "backend.src.processing.utils.processing_utils.generate_document_summary",
                        return_value=("This is a summary", [0.1] * 384),
                    ):
                        with patch.object(
                            pipeline.embedder_service,
                            "get_embedder",
                            return_value=MagicMock(
                                embed_source=AsyncMock(return_value=mock_source)
                            ),
                        ):
                            result = await pipeline.process_document(
                                file_path=sample_markdown_path,
                                name="test.md",
                                ctx=ctx,
                            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_document_with_vector_storage(
        self,
        mock_session: AsyncMock,
        mock_vector_service: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test document processing with vector storage."""
        ctx = mock_request_context
        pipeline = ProcessingPipeline(mock_session, mock_vector_service)

        mock_parsed = ParsedDocument(
            content="Test content",
            page_count=1,
            parser_used="MARKDOWN",
            errors=[],
        )

        mock_chunk = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test content",
            start_index=0,
            end_index=12,
            token_count=2,
            dense_vector=[0.1] * 384,
        )

        mock_source = SourceWithRelations(
            source_id="source_001",
            source_hash="hash123",
            unique_id="unique_001",
            source_content="Test content",
            source_summary="",
            summary_embedding="",
            workspace_ids=[ctx.workspace_id],
            chunks=[mock_chunk],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker_instance = MagicMock()
                mock_chunker_instance.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker_instance

                with patch.object(
                    pipeline.embedder_service,
                    "get_embedder",
                    return_value=MagicMock(
                        embed_source=AsyncMock(return_value=mock_source)
                    ),
                ):
                    result = await pipeline.process_document(
                        file_path=sample_markdown_path, name="test.md", ctx=ctx
                    )

        assert result.success is True
        mock_vector_service.upsert.assert_called_once()


class TestProcessDocumentErrors:
    """Tests for error handling in document processing."""

    @pytest.mark.asyncio
    async def test_process_document_parse_error(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test handling of parsing errors."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_parsed = ParsedDocument(
            content="",
            page_count=0,
            parser_used="MARKDOWN",
            errors=["Failed to parse document"],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            result = await pipeline.process_document(
                file_path=sample_markdown_path, name="test.md", ctx=ctx
            )

        assert result.success is False
        assert "Failed to parse document" in result.errors

    @pytest.mark.asyncio
    async def test_process_document_no_embeddings(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test handling when no chunks are embedded."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_parsed = ParsedDocument(
            content="Test content",
            page_count=1,
            parser_used="MARKDOWN",
            errors=[],
        )

        # Chunk without embedding
        mock_chunk = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test content",
            start_index=0,
            end_index=12,
            token_count=2,
            dense_vector=None,
        )

        mock_source = SourceWithRelations(
            source_id="source_001",
            source_hash="hash123",
            unique_id="unique_001",
            source_content="Test content",
            source_summary="",
            summary_embedding="",
            workspace_ids=[ctx.workspace_id],
            chunks=[mock_chunk],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker_instance = MagicMock()
                mock_chunker_instance.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker_instance

                with patch.object(
                    pipeline.embedder_service,
                    "get_embedder",
                    return_value=MagicMock(
                        embed_source=AsyncMock(return_value=mock_source)
                    ),
                ):
                    result = await pipeline.process_document(
                        file_path=sample_markdown_path, name="test.md", ctx=ctx
                    )

        assert result.success is False
        assert "No chunks were successfully embedded" in result.errors

    @pytest.mark.asyncio
    async def test_process_document_partial_embeddings(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test warning when only some chunks are embedded."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_parsed = ParsedDocument(
            content="Test content",
            page_count=1,
            parser_used="MARKDOWN",
            errors=[],
        )

        # One embedded, one not
        mock_chunk1 = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test content 1",
            start_index=0,
            end_index=14,
            token_count=3,
            dense_vector=[0.1] * 384,
        )
        mock_chunk2 = Chunk(
            chunk_id="chunk_002",
            source_id="source_001",
            source_hash="hash123",
            text="Test content 2",
            start_index=15,
            end_index=29,
            token_count=3,
            dense_vector=None,
        )

        mock_source = SourceWithRelations(
            source_id="source_001",
            source_hash="hash123",
            unique_id="unique_001",
            source_content="Test content",
            source_summary="",
            summary_embedding="",
            workspace_ids=[ctx.workspace_id],
            chunks=[mock_chunk1, mock_chunk2],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ):
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker_instance = MagicMock()
                mock_chunker_instance.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker_instance

                with patch.object(
                    pipeline.embedder_service,
                    "get_embedder",
                    return_value=MagicMock(
                        embed_source=AsyncMock(return_value=mock_source)
                    ),
                ):
                    result = await pipeline.process_document(
                        file_path=sample_markdown_path, name="test.md", ctx=ctx
                    )

        assert result.success is True
        assert len(result.warnings) == 1
        assert "1/2 chunks embedded" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_process_document_exception(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test handling of unexpected exceptions."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        with patch.object(
            pipeline.parser_service,
            "parse_document",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await pipeline.process_document(
                file_path=sample_markdown_path, name="test.md", ctx=ctx
            )

        assert result.success is False
        assert "Unexpected error" in result.errors
        assert result.processing_time_ms > 0  # Time should still be recorded


# ============================================================================
# ProcessingPipeline.process_batch Tests
# ============================================================================


class TestProcessBatch:
    """Tests for batch document processing."""

    @pytest.mark.asyncio
    async def test_process_batch_single_document(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test batch processing with single document."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_result = ProcessingResult(
            source=MagicMock(),
            chunk_count=1,
            errors=[],
        )

        with patch.object(
            pipeline, "process_document", return_value=mock_result
        ) as mock_process:
            results = await pipeline.process_batch(
                [(sample_markdown_path, "test.md")], ctx=ctx
            )

        assert len(results) == 1
        assert results[0].success is True
        mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_multiple_documents(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        tmp_path: Path,
    ):
        """Test batch processing with multiple documents."""
        ctx = mock_request_context
        # Create multiple test files
        docs = []
        for i in range(3):
            path = tmp_path / f"doc_{i}.md"
            path.write_text(f"# Document {i}\n\nContent for document {i}.")
            docs.append((str(path), f"doc_{i}.md"))

        pipeline = ProcessingPipeline(mock_session)

        call_count = 0

        async def mock_process(file_path, name, ctx):
            nonlocal call_count
            call_count += 1
            return ProcessingResult(
                source=MagicMock(),
                chunk_count=1,
                errors=[],
            )

        with patch.object(pipeline, "process_document", side_effect=mock_process):
            results = await pipeline.process_batch(docs, ctx=ctx)

        assert len(results) == 3
        assert call_count == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_respects_concurrency(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        tmp_path: Path,
    ):
        """Test that batch processing respects max_concurrent limit."""
        ctx = mock_request_context
        # Create test files
        docs = []
        for i in range(5):
            path = tmp_path / f"doc_{i}.md"
            path.write_text(f"# Document {i}")
            docs.append((str(path), f"doc_{i}.md"))

        pipeline = ProcessingPipeline(mock_session)

        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_process(file_path, name, ctx):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.05)  # Simulate processing time
            concurrent_count -= 1
            return ProcessingResult(source=MagicMock(), chunk_count=1, errors=[])

        with patch.object(pipeline, "process_document", side_effect=mock_process):
            results = await pipeline.process_batch(docs, ctx=ctx, max_concurrent=2)

        assert len(results) == 5
        assert max_concurrent_observed <= 2

    @pytest.mark.asyncio
    async def test_process_batch_preserves_order(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        tmp_path: Path,
    ):
        """Test that batch results maintain input order."""
        ctx = mock_request_context
        docs = []
        for i in range(4):
            path = tmp_path / f"doc_{i}.md"
            path.write_text(f"# Document {i}")
            docs.append((str(path), f"doc_{i}.md"))

        pipeline = ProcessingPipeline(mock_session)

        async def mock_process(file_path, name, ctx):
            # Simulate varying processing times
            idx = int(name.split("_")[1].split(".")[0])
            await asyncio.sleep(0.01 * (4 - idx))  # Reverse order timing
            result = ProcessingResult(source=MagicMock(), chunk_count=idx, errors=[])
            return result

        with patch.object(pipeline, "process_document", side_effect=mock_process):
            results = await pipeline.process_batch(docs, ctx=ctx)

        # Verify order is preserved (chunk_count matches index)
        for i, result in enumerate(results):
            assert result.chunk_count == i

    @pytest.mark.asyncio
    async def test_process_batch_empty_list(
        self,
        mock_session: AsyncMock,
        mock_request_context,
    ):
        """Test batch processing with empty document list."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        results = await pipeline.process_batch([], ctx=ctx)

        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_mixed_success_failure(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        tmp_path: Path,
    ):
        """Test batch processing with mixed success and failure."""
        ctx = mock_request_context
        docs = []
        for i in range(3):
            path = tmp_path / f"doc_{i}.md"
            path.write_text(f"# Document {i}")
            docs.append((str(path), f"doc_{i}.md"))

        pipeline = ProcessingPipeline(mock_session)

        async def mock_process(file_path, name, ctx):
            if "doc_1" in name:
                return ProcessingResult(source=None, errors=["Failed"])
            return ProcessingResult(source=MagicMock(), chunk_count=1, errors=[])

        with patch.object(pipeline, "process_document", side_effect=mock_process):
            results = await pipeline.process_batch(docs, ctx=ctx)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# ============================================================================
# Private Helper Method Tests
# ============================================================================


class TestParseDocument:
    """Tests for _parse_document helper method."""

    @pytest.mark.asyncio
    async def test_parse_document_with_config(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test parsing with custom config."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context
        ctx.preferences.parser.parser_type = ParserType.MARKDOWN
        ctx.preferences.parser.max_pages = 5

        mock_parsed = ParsedDocument(
            content="Test", page_count=1, parser_used="MARKDOWN", errors=[]
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ) as mock_parse:
            result = await pipeline._parse_document(sample_markdown_path, ctx)

        mock_parse.assert_called_once_with(
            file_path=sample_markdown_path,
            config=ctx.preferences.parser,
            preferred_parser=ParserType.MARKDOWN,
            ctx=ctx,
        )
        assert result == mock_parsed

    @pytest.mark.asyncio
    async def test_parse_document_default_config(
        self,
        mock_session: AsyncMock,
        mock_request_context,
        sample_markdown_path: str,
    ):
        """Test parsing with default config."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_parsed = ParsedDocument(
            content="Test", page_count=1, parser_used="MARKDOWN", errors=[]
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ) as mock_parse:
            await pipeline._parse_document(sample_markdown_path, ctx)

        mock_parse.assert_called_once()
        assert "config" in mock_parse.call_args.kwargs


class TestCreateSourceWithChunks:
    """Tests for _create_source_with_chunks helper method."""

    @pytest.mark.asyncio
    async def test_create_source_with_chunks(
        self,
        mock_session: AsyncMock,
        parsed_document: ParsedDocument,
        mock_request_context,
    ):
        """Test source and chunk creation."""
        pipeline = ProcessingPipeline(mock_session)
        # RequestContext is frozen, create a new one
        ctx = mock_request_context.with_new_workspace_id(42)

        mock_chunk = Chunk(
            chunk_id="chunk_001",
            source_id="source_001",
            source_hash="hash123",
            text="Test",
            start_index=0,
            end_index=4,
            token_count=1,
        )

        with patch(
            "backend.src.processing.utils.processing_utils.generate_content_hash",
            return_value="content_hash_123",
        ):
            with patch(
                "backend.src.processing.utils.processing_utils.generate_unique_identifier_hash",
                return_value="unique_hash_123",
            ):
                with patch(
                    "backend.src.processing.services.chunker_service.ChunkerService"
                ) as MockChunker:
                    mock_source = SourceWithRelations(
                        source_id="source_001",
                        source_hash="content_hash_123",
                        unique_id="unique_hash_123",
                        source_content=parsed_document.content,
                        source_summary="",
                        summary_embedding="",
                        workspace_ids=[42],
                        chunks=[mock_chunk],
                    )
                    mock_chunker_instance = MagicMock()
                    mock_chunker_instance.chunk_source = AsyncMock(
                        return_value=mock_source
                    )
                    MockChunker.return_value = mock_chunker_instance

                    result = await pipeline._create_source_with_chunks(
                        "test.md", parsed_document, ctx
                    )

        assert result.source_hash == "content_hash_123"
        assert result.unique_id == "unique_hash_123"
        assert result.workspace_ids == [42]
        assert len(result.chunks) == 1


class TestGenerateSummary:
    """Tests for _generate_summary helper method."""

    @pytest.mark.asyncio
    async def test_generate_summary_success(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test successful summary generation."""
        pipeline = ProcessingPipeline(mock_session)
        # RequestContext is frozen, create a new one
        ctx = RequestContext.create_simple(
            user_id="user123",
            workspace_id=1,
            preferences=mock_request_context.preferences,
        )

        with patch(
            "backend.src.convo.services.llm_service.LLMService.get_llm_for_role",
            return_value=MagicMock(),
        ):
            with patch(
                "backend.src.processing.utils.processing_utils.generate_document_summary",
                return_value=("Generated summary", [0.5] * 10),
            ):
                result = await pipeline._generate_summary(sample_source, ctx)

        assert result.source_summary == "Generated summary"
        assert result.summary_embedding != ""

    @pytest.mark.asyncio
    async def test_generate_summary_no_ctx(
        self, mock_session: AsyncMock, sample_source: SourceWithRelations
    ):
        """Test summary generation skipped without context."""
        pipeline = ProcessingPipeline(mock_session)

        result = await pipeline._generate_summary(source=sample_source, ctx=None)  # pyright: ignore

        # Should return source unchanged
        assert result.source_summary == ""

    @pytest.mark.asyncio
    async def test_generate_summary_exception_handling(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test summary generation handles exceptions gracefully."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        with patch(
            "backend.src.convo.services.llm_service.LLMService.get_llm_for_role",
            side_effect=Exception("LLM error"),
        ):
            # Should not raise, just log warning
            result = await pipeline._generate_summary(sample_source, ctx)

        assert result.source_summary == ""

    @pytest.mark.asyncio
    async def test_generate_summary_no_user_id(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test summary generation skipped without user_id in context."""
        pipeline = ProcessingPipeline(mock_session)
        # RequestContext is frozen, create a new one with empty user_id
        ctx = RequestContext.create_simple(
            user_id="",
            workspace_id=mock_request_context.workspace_id,
            preferences=mock_request_context.preferences,
        )

        result = await pipeline._generate_summary(sample_source, ctx)

        # Should return source unchanged
        assert result.source_summary == ""


class TestEmbedSource:
    """Tests for _embed_source helper method."""

    @pytest.mark.asyncio
    async def test_embed_source_success(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test successful embedding."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        embedded_source = sample_source
        embedded_source.chunks[0].dense_vector = [0.1] * 384

        mock_embedder = MagicMock()
        mock_embedder.embed_source = AsyncMock(return_value=embedded_source)

        with patch.object(
            pipeline.embedder_service, "get_embedder", return_value=mock_embedder
        ):
            result = await pipeline._embed_source(sample_source, ctx)

        assert result.chunks[0].dense_vector is not None
        mock_embedder.embed_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_source_no_embedder(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test embedding when no embedder is available."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        with patch.object(pipeline.embedder_service, "get_embedder", return_value=None):
            result = await pipeline._embed_source(sample_source, ctx)

        # Should return source unchanged
        assert result == sample_source

    @pytest.mark.asyncio
    async def test_embed_source_default_config(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test embedding uses default config when none provided."""
        pipeline = ProcessingPipeline(mock_session)
        ctx = mock_request_context

        mock_embedder = MagicMock()
        mock_embedder.embed_source = AsyncMock(return_value=sample_source)

        with patch.object(
            pipeline.embedder_service, "get_embedder", return_value=mock_embedder
        ):
            await pipeline._embed_source(sample_source, ctx)

        mock_embedder.embed_source.assert_called_once_with(sample_source, ctx=ctx)


class TestStoreVectors:
    """Tests for _store_vectors helper method."""

    @pytest.mark.asyncio
    async def test_store_vectors_success(
        self,
        mock_session: AsyncMock,
        mock_vector_service: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test successful vector storage."""
        pipeline = ProcessingPipeline(mock_session, mock_vector_service)
        ctx = mock_request_context

        await pipeline._store_vectors(sample_source, ctx)

        mock_vector_service.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_vectors_no_repo(
        self,
        mock_session: AsyncMock,
        sample_source: SourceWithRelations,
        mock_request_context,
    ):
        """Test vector storage skipped when no repo configured."""
        pipeline = ProcessingPipeline(mock_session, vector_service=None)

        # Should complete without error
        await pipeline._store_vectors(sample_source, mock_request_context)


# ============================================================================
# Integration Tests
# ============================================================================


class TestProcessingPipelineIntegration:
    """Integration tests for the processing pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_markdown(
        self, mock_session: AsyncMock, sample_markdown_path: str, mock_request_context
    ):
        """Test complete pipeline with real markdown file."""
        ctx = mock_request_context
        ctx.preferences.parser.parser_type = ParserType.MARKDOWN
        ctx.preferences.parser.generate_summary = False
        ctx.preferences.vector_store = None

        pipeline = ProcessingPipeline(mock_session)

        # Mock only the embedder to avoid needing actual model
        mock_embedder = MagicMock()

        async def mock_embed(source, ctx):
            for chunk in source.chunks:
                chunk.dense_vector = [0.1] * 384
            return source

        mock_embedder.embed_source = mock_embed

        with patch.object(
            pipeline.embedder_service, "get_embedder", return_value=mock_embedder
        ):
            result = await pipeline.process_document(
                file_path=sample_markdown_path, name="test.md", ctx=ctx
            )

        assert result.success is True
        assert result.chunk_count > 0
        assert result.page_count >= 1
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_config_propagation(
        self, mock_session: AsyncMock, sample_markdown_path: str, mock_request_context
    ):
        """Test that config is properly propagated through pipeline."""
        # RequestContext is frozen, update preferences then create new context
        prefs = mock_request_context.preferences
        prefs.parser.parser_type = ParserType.MARKDOWN
        prefs.parser.max_pages = 10
        prefs.embedder.embedder_type = EmbedderType.FASTEMBED
        prefs.embedder.batch_size = 32
        prefs.parser.generate_summary = False
        prefs.vector_store = None  # type: ignore

        ctx = RequestContext.create_simple(
            user_id=mock_request_context.user_id, workspace_id=99, preferences=prefs
        )

        pipeline = ProcessingPipeline(mock_session)

        mock_parsed = ParsedDocument(
            content="Test", page_count=1, parser_used="MARKDOWN", errors=[]
        )

        mock_chunk = Chunk(
            chunk_id="c1",
            source_id="s1",
            source_hash="h1",
            text="Test",
            start_index=0,
            end_index=4,
            token_count=1,
            dense_vector=[0.1] * 384,
        )

        mock_source = SourceWithRelations(
            source_id="s1",
            source_hash="h1",
            unique_id="u1",
            source_content="Test",
            source_summary="",
            summary_embedding="",
            workspace_ids=[99],
            chunks=[mock_chunk],
        )

        with patch.object(
            pipeline.parser_service, "parse_document", return_value=mock_parsed
        ) as mock_parse:
            with patch(
                "backend.src.processing.services.chunker_service.ChunkerService"
            ) as MockChunker:
                mock_chunker = MagicMock()
                mock_chunker.chunk_source = AsyncMock(return_value=mock_source)
                MockChunker.return_value = mock_chunker

                with patch.object(
                    pipeline.embedder_service, "get_embedder"
                ) as mock_get_embedder:
                    mock_embedder = MagicMock()
                    mock_embedder.embed_source = AsyncMock(return_value=mock_source)
                    mock_get_embedder.return_value = mock_embedder

                    await pipeline.process_document(
                        file_path=sample_markdown_path, name="test.md", ctx=ctx
                    )

        # Verify parser config was passed
        mock_parse.assert_called_once()
        assert mock_parse.call_args.kwargs["config"] == ctx.preferences.parser
        assert mock_parse.call_args.kwargs["preferred_parser"] == ParserType.MARKDOWN

        # Verify embedder type was used
        mock_get_embedder.assert_called_with(EmbedderType.FASTEMBED)
