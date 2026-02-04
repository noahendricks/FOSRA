from typing import Any

from msgspec import field

from FOSRABack.src.storage.utils.converters import DomainStruct
from typing import TYPE_CHECKING

from FOSRABack.src.domain.schemas.source_schemas import PayloadShape

# ============================================================================
# Vectors
# ============================================================================


class VectorPoint(DomainStruct):
    id: str
    payload: PayloadShape
    dense_vector: list[float] | None = None
    late_interaction_vectors: list[list[float]] | None = None


class EmbeddingResult(DomainStruct):
    """Result of embedding operation."""

    dense_vectors: list[list[float]]
    late_interaction_vectors: list[list[list[float]]] | None = None
    embedder_used: str = ""
    embed_time_ms: float = 0.0
    token_count: int = 0
    errors: list[str] = field(default_factory=list)


class VectorSearchResult(DomainStruct):
    """Standardized result from vector search."""

    query_text: str
    point_id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None


class ParsedDocument(DomainStruct):
    """Result of document parsing."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int = 1
    parser_used: str = ""
    parse_time_ms: float = 0.0
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class StreamValidationResult(DomainStruct):
    """Result of stream request validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    remediation: str | None = None
    warnings: list[str] = field(default_factory=list)


class StreamingStats(DomainStruct):
    """Streaming statistics."""

    active_streams: int
    total_chunks_sent: int
    average_stream_duration_seconds: float
    success_rate: float
    error_count: int
    last_updated: str
