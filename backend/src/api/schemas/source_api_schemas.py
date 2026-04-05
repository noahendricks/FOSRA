from datetime import datetime
from typing import Any

from pydantic import Field

from backend.src.api.schemas.base import BaseModelFlex
from backend.src.domain.enums import DocType, SourceType
from backend.src.storage.utils.converters import utc_now

# ============================================================================
# File
# API INPUT SCHEMAS (Request DTOs)
# ============================================================================


# ============================================================================
# Source
# API OUTPUT SCHEMAS (Response DTOs)
# ============================================================================
#
#
class ChunkResponse(BaseModelFlex):
    chunk_id: str
    source_id: str
    source_hash: str
    start_index: int
    end_index: int
    token_count: int
    text: str


class SourceRequest(BaseModelFlex):
    """Response DTO for a source in a directory session."""

    source_id: str


class SourceResponseBase(BaseModelFlex):
    """Response DTO for a source in a directory session."""

    id: str
    type: SourceType | str | None
    hash: str | None = None  # WARN: Change this to Non-Nullable
    name: str = ""
    document_type: DocType | None = None
    source_summary: str = ""
    summary_embedding: str = ""
    uploaded_at: datetime = Field(default_factory=utc_now)


class SourceResponseShallow(BaseModelFlex):
    """Response DTO for a source in a directory session."""

    id: str
    name: str = ""
    source_summary: str = ""
    source_type: SourceType
    summary_embedding: str = ""
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now())


class SourceResponseDeep(SourceResponseBase):
    """Response DTO for a source in a directory session."""

    metadata: dict[str, Any] = {}
    result_score: float = 0.0


class ChunkWithScoreResponse(BaseModelFlex):
    """Chunk with retrieval scoring."""

    chunk: ChunkResponse
    similarity_score: float
    reranker_score: float | None = None


class SourceGroupResponse(BaseModelFlex):
    """Grouped source with chunks for UI display."""

    source: SourceResponseDeep
    chunks: list[ChunkWithScoreResponse]
    top_score: float
    chunk_count: int
