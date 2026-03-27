"""
Typed TUI event envelope for FastAPI SSE.

All server→TUI events use this envelope:
    event: <type>
    data: {"properties": {...}}

Wraps the raw event dict published by subsystems into a typed structure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TUIEvent(BaseModel):
    type: str
    properties: dict[str, Any]
    sequence_nr: int | None = None
    retry: int = 5000

    def to_sse_data(self) -> dict[str, Any]:
        return {"properties": self.properties}

    @property
    def event_type(self) -> str:
        return self.type
