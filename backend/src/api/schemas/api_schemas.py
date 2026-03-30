from __future__ import annotations

from datetime import datetime
from typing import Any

import msgspec
from pydantic import Field, field_serializer

from backend.src.api.schemas.base import _BaseModelFlex
from backend.src.api.schemas.tui_control_schemas import (  # noqa: F401
    FilePart,
    TextPart,
    UIMessage,
    UIMessagePart,
)
from backend.src.domain.enums import MessageRole
from backend.src.storage.utils.converters import utc_now


class MessageRequest(_BaseModelFlex):
    user_id: str
    convo_id: str
    message_id: str
    parent_message_id: str | None = None
    root_message_id: str | None = None
    role: MessageRole | str
    messages: list[UIMessage] = []
    text: str | None = None
    trigger: str
    attached_files: list[FilePart] = []
    message_metadata: dict[str, Any] | None = None


class MessageResponse(_BaseModelFlex):
    role: MessageRole
    user_id: str | None = None
    convo_id: str
    text: str
    message_id: str | None = None
    parent_id: str | None = None
    root_id: str | None = None
    message_metadata: dict[str, Any] | None = None
    attached_files: list[dict[str, Any]] | None = None
    attached_sources: list[dict[str, Any]] | None = []
    sources_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: utc_now())

    @field_serializer("attached_sources")
    def serialize_sources(self, sources: list[Any] | None) -> list[dict[str, Any]]:
        if not sources:
            return []

        result = []
        for source in sources:
            if hasattr(source, "model_dump"):
                result.append(source.model_dump(mode="json"))
            elif isinstance(source, dict):
                result.append(source)
            else:
                result.append(msgspec.to_builtins(source))
        return result

    def to_litellm_format(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.text}


class MessageUpdateRequest(MessageRequest):
    message_id: str
    message_metadata: dict[str, Any] | None = None
