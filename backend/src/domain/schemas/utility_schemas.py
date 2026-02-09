from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from msgspec import field
from backend.src.storage.utils.converters import (
    DomainStruct,
    utc_now,
)


if TYPE_CHECKING:
    pass


# ============================================================================
# UTILITY / HELPER SCHEMAS
# ============================================================================


class ProcessingStatus(DomainStruct):
    """Status update during source processing."""

    total_sources: int
    processed_sources: int
    embedded_sources: int
    failed_sources: int

    current_source: str | None = None
    current_stage: str | None = None

    errors: list[str] = field(default_factory=list)


class HealthCheck(DomainStruct):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = field(default_factory=utc_now)

    services: dict[str, str] = field(
        default_factory=dict,
    )


class PaginatedResponseOut(DomainStruct):
    """Generic paginated response."""

    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 50
    total_pages: int = 1
