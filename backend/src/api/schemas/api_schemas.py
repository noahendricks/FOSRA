from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import msgspec
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from pydantic.v1.utils import to_camel

from backend.src.api.schemas.source_api_schemas import (
    SourceGroupResponse,
    SourceResponseDeep,
    SourceResponseShallow,
)
from backend.src.domain.enums import MessageRole
from backend.src.storage.utils.converters import utc_now


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG


class MessageAPI(_BaseModelFlex):
    text: str
    convo_id: str
    role: MessageRole
    user_id: str | None = None
    message_id: str | None = None
    message_metadata: dict[str, Any] | None = None


class TextPart(_BaseModelFlex):
    type: str
    text: str


class FilePart(_BaseModelFlex):
    type: str
    name: str
    size: int
    filename: str
    bytes: bytes
    media_type: str
    url: str | None = None


UIMessagePart = TextPart | FilePart


class UIMessage(_BaseModelFlex):
    id: str
    role: str
    parts: list[UIMessagePart]
    message_metadata: dict[str, Any] | None = None


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


class ConvoRequestBase(_BaseModelFlex):
    user_id: str


class ConvoRequest(ConvoRequestBase):
    convo_id: str
    pass


class NewConvoRequest(ConvoRequestBase):
    title: str | None = "New Convo"


class ConvoDeleteRequest(ConvoRequestBase):
    convo_id: str
    convo_list: list[str] | None = None


class ConvoUpdateRequest(ConvoRequestBase):
    title: str | None = "New Convo"
    convo_id: str
    convo_metadata: dict[str, Any] | None = None
    messages: list[MessageAPI] | None = None
    data: dict[str, Any] | None = None


class ConvoListItemResponse(ConvoRequestBase):
    title: str
    convo_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = Field(default=0)


class NewConvoResponse(ConvoRequestBase):
    title: str | None = "New Convo"
    convo_metadata: dict[str, Any] | None = None


class ConvoFullResponse(ConvoRequestBase):
    convo_id: str
    title: str | None = "New Convo"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = 0

    knowledge_sources: list[SourceResponseDeep | SourceResponseShallow] = Field(
        default_factory=list,
        description="All sources for this chat",
    )

    messages: list[MessageResponse] = Field(default_factory=list)


class CompletionResponse(_BaseModelFlex):
    message: MessageAPI
    usage: str = Field(
        description="Token usage: {'input_tokens': X, 'output_tokens': Y, 'total_tokens': Z}"
    )
    retrieval_time_ms: int | None = Field(
        default=None, description="Time spent on RAG retrieval in milliseconds"
    )

    finish_reason: str | None = Field(
        default=None,
        description="Why generation stopped: 'end_turn', 'max_tokens', 'stop_sequence'",
    )


class StreamChunkResponse(_BaseModelFlex):
    type: Literal["content", "rag_sources", "done"]
    delta: str | None = None
    rag_source: list[SourceGroupResponse] | None = None
    usage: str | None = None
    retrieval_time_ms: int | None = None
