from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.v1.utils import to_camel

from backend.src.domain.enums import DocType, FileType, SourceType
from backend.src.storage.utils.converters import utc_now

# ============================================================================
# File
# API INPUT SCHEMAS (Request DTOs)
# ============================================================================


class _BaseModelFlex(BaseModel):
    """_BaseModelFlex with flexible config for attribute-based initialization."""

    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config = _FLEXIBLE_CONFIG


# ============================================================================
# Source
# API OUTPUT SCHEMAS (Response DTOs)
# ============================================================================
#
#
class ChunkResponse(_BaseModelFlex):
    chunk_id: str
    source_id: str
    source_hash: str
    start_index: int
    end_index: int
    token_count: int
    text: str


class SourceRequest(_BaseModelFlex):
    """Response DTO for a source in a directory session."""

    source_id: str


class SourceResponseBase(_BaseModelFlex):
    """Response DTO for a source in a directory session."""

    id: str
    type: SourceType | str | None
    hash: str | None = None  # WARN: Change this to Non-Nullable
    name: str = ""
    document_type: DocType | None = None
    source_summary: str = ""
    summary_embedding: str = ""
    uploaded_at: datetime = Field(default_factory=utc_now)


class SourceResponseShallow(_BaseModelFlex):
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


class ChunkWithScoreResponse(_BaseModelFlex):
    """Chunk with retrieval scoring."""

    chunk: ChunkResponse
    similarity_score: float
    reranker_score: float | None = None


class SourceGroupResponse(_BaseModelFlex):
    """Grouped source with chunks for UI display."""

    source: SourceResponseDeep
    chunks: list[ChunkWithScoreResponse]
    top_score: float
    chunk_count: int
