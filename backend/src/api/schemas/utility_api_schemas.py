# ============================================================================
# UTILITY / HELPER SCHEMAS
# ============================================================================


from datetime import datetime
from typing import Any, Literal

from pydantic.v1.utils import to_camel

from backend.src.storage.utils.converters import utc_now


from pydantic import BaseModel, Field, ConfigDict


class _BaseModelFlex(BaseModel):
    """_BaseModelFlex with flexible config for attribute-based initialization."""

    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config = _FLEXIBLE_CONFIG


class ProcessingStatusResponse(_BaseModelFlex):
    """Status update during source processing."""

    total_sources: int
    processed_sources: int
    embedded_sources: int
    failed_sources: int

    current_source: str | None = None
    current_stage: str | None = None

    errors: list[str] = Field(default_factory=list)


class HealthCheckResponse(_BaseModelFlex):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=utc_now)

    services: dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependencies (DB, Qdrant, LLM provider)",
    )


class PaginatedResponse(_BaseModelFlex):
    """Generic paginated response."""

    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 50
    total_pages: int = 1
