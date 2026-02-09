from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any


from msgspec import field

from backend.src.domain.enums import (
    DocumentType,
    FileType,
    RetrievalMode,
    SearchStrategy,
)
from backend.src.storage.utils.converters import DomainStruct, utc_now

from backend.src.domain.enums import OriginType

if TYPE_CHECKING:
    from backend.src.domain.enums import OriginType


# ============================================================================
#  Source Schemas
# ============================================================================


class Source(DomainStruct):
    origin_path: str
    origin_type: OriginType | None = None
    source_id: str = ""
    source_type: FileType | None = None
    uploaded_at: datetime = field(default_factory=lambda: datetime.now())
    source_size: int | None = None
    hash: str | None = None
    name: str = ""
    document_type: DocumentType | None = None
    content: str | None = None
    source_summary: str = ""
    summary_embedding: str = ""


class SourceProcessing(Source):
    page_count: int = 1
    parser_used: str = ""
    parse_time_ms: float = 0.0
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SourceUpdate(Source):
    """Properties for updating a source."""
    source_summary: str | None = None #pyright: ignore
    summary_embedding: str | None = None #pyright: ignore


class SourceFull(Source):
    """Source schema for API responses."""

    workspace_ids: list[int] = field(default_factory=list)
    chunks: list[ChunkFull] = []
    summary_embedding: str = ""
    content: str | None = None
    metadata: dict[str, Any] = {}


# ============================================================================
# Chunk Schemas
# ============================================================================


class Chunk(DomainStruct):
    chunk_id: str
    source_id: str
    source_hash: str
    start_index: int
    end_index: int
    token_count: int
    text: str


class NewChunk(Chunk):
    """Schema for creating a chunk (before embeddings)."""

    source_id: str
    source_hash: str


class ChunkFull(Chunk):
    dense_vector: list[float] | None = field(default=None)
    qdrant_point_id: str | None = None


class ChunkingResult(DomainStruct):
    """Result of chunking operation."""

    chunks: list[ChunkFull]
    chunker_used: str = ""
    chunk_time_ms: float = 0.0
    total_tokens: int = 0
    errors: list[str] = field(default_factory=list)


class ChunkWithScore(DomainStruct):
    """Chunk with retrieval scoring."""

    chunk: ChunkFull
    similarity_score: float
    reranker_score: float | None = None


class SourceGroup(DomainStruct):
    """Grouped source with chunks for UI display."""

    source: SourceFull
    chunks: list[ChunkWithScore]

    top_score: float
    chunk_count: int


class PayloadShape(DomainStruct):
    chunk_id: str
    name: str
    source_id: str
    source_name: str
    source_hash: str
    origin_type: str
    origin_path: str
    file_type: str | None
    chunk_text: str
    start_index: int
    end_index: int
    token_count: int


class RetrievedResult(DomainStruct):
    """Schema for retrieved context from RAG."""

    chunk_id: str
    query_text: str
    source_id: str
    source_name: str | None = None
    file_type: FileType | None = None
    origin_type: str | None = None

    similarity_score: float = 0.0
    result_rank: int | None = None
    reranker_score: float | None = None

    retrieved_at: datetime = field(default_factory=utc_now)

    model_used: str | None = None

    source_snippet: str | None = field(
        default=None,
    )

    contents: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format this retrieved context as a string for prompt construction."""

        rank = self.result_rank if self.result_rank is not None else "N/A"

        return f"[#{rank} RANKED CHUNK: FILE NAME: {self.source_name} CHUNK CONTENT: {self.contents}]"


class RerankResult(DomainStruct):
    """Result of reranking operation."""

    documents: list[RetrievedResult]
    reranker_used: str = ""
    rerank_time_ms: float = 0.0
    original_count: int = 0
    filtered_count: int = 0
    errors: list[str] = field(default_factory=list)


class RetrievalConfig(DomainStruct):
    """Configuration for retrieval operations."""

    # Basic settings
    top_k: int = 10
    min_score: float = 0.0
    mode: RetrievalMode = RetrievalMode.CHUNKS
    strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY

    # Filtering
    file_types: list[FileType] | None = None
    source_ids: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None

    # Reranking
    enable_rerank: bool = False
    rerank_top_k: int | None = None

    # Advanced options
    include_content: bool = True
    include_metadata: bool = True
    deduplicate: bool = True






