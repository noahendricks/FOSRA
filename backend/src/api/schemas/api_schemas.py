from __future__ import annotations
import msgspec

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_serializer
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

    model_config: ConfigDict = _FLEXIBLE_CONFIG #pyright: ignore


class MessageAPI(_BaseModelFlex):
    """Base message properties."""

    text: str
    convo_id: str
    role: MessageRole
    user_id: str | None = None
    message_id: str | None = None
    message_metadata: dict[str, Any] | None = None


# ============================================================================
# User Schemas
# ============================================================================


class UserRequestBase(_BaseModelFlex):
    """Base user properties."""

    username: str
    password: str
    enabled: bool = True


class UserLogin(_BaseModelFlex):
    """Base user properties."""

    username: str
    password: str
    enabled: bool = True


class NewUserRequest(UserLogin):
    pass


class UserRequest(_BaseModelFlex):
    user_id: str
    created_at: datetime = Field(default_factory=utc_now)


class UserResponse(_BaseModelFlex):
    user_id: str
    username: str
    created_at: datetime = Field(default_factory=utc_now)
    last_login: datetime = Field(default_factory=utc_now)


class NewUserResponse(_BaseModelFlex):
    user_id: str


# ============================================================================
# Workspace
# API INPUT SCHEMAS (Request DTOs)
# ============================================================================
#
class NewWorkspaceRequest(_BaseModelFlex):
    name: str = Field(default="New Workspace", max_length=100)
    user_id: str
    description: str | None = ""


class WorkspaceRequest(_BaseModelFlex):
    workspace_id: str
    user_id: str


class WorkspaceDeleteRequest(WorkspaceRequest):
    workspace_list: list[int] = []


class WorkspaceUpdateRequest(WorkspaceRequest):
    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)


# ============================================================================
# API OUTPUT SCHEMAS (Response DTOs)
# ============================================================================
#
class NewWorkspaceResponse(_BaseModelFlex):
    name: str = Field(default="New Workspace", max_length=100)
    user_id: str
    workspace_id: str


class WorkspaceFullResponse(WorkspaceRequest):
    name: str
    sources_count: int = 0
    conversations_count: int = 0
    description: str | None = Field(default=None, max_length=500)
    archived_convos: list[str] | None = None


# ============================================================================
# Message Schemas
# ============================================================================
#


class TextPart(_BaseModelFlex):
    """A Text part of a UI message (matches AI SDK format)"""

    type: str
    text: str


class FilePart(_BaseModelFlex):
    """A File part of a UI message (matches AI SDK format)"""

    type: str
    name: str
    size: int
    filename: str
    bytes: bytes
    media_type: str
    url: str | None = None


UIMessagePart = TextPart | FilePart


class UIMessage(_BaseModelFlex):
    """A File part of a UI message (matches AI SDK format)"""

    id: str
    role: str
    parts: list[UIMessagePart]
    message_metadata: dict[str, Any] | None = None


# NOTE: Everything necessary for User Message -> Assistant Message
class MessageRequest(_BaseModelFlex):
    """Request body for sending a (User) message"""

    user_id: str
    workspace_id: str
    convo_id: str
    message_id: str
    role: MessageRole | str
    messages: list[UIMessage] = []
    text: str | None = None
    trigger: str  # 'submit-message' | 'regenerate-message'
    attached_files: list[FilePart] = []
    message_metadata: dict[str, Any] | None = None


class MessageResponse(_BaseModelFlex):
    """Response for User Messages and Assistant Messages"""

    role: MessageRole
    user_id: str | None = None
    convo_id: str
    text: str
    message_id: str | None = None
    message_metadata: dict[str, Any] | None = None
    attached_files: list[dict[str, Any]] | None = None
    attached_sources: list[dict[str, Any]] | None = []
    sources_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: utc_now())

    @field_serializer("attached_sources")
    def serialize_sources(self, sources: list[Any] | None) -> list[dict[str, Any]]:
        """Ensure sources are always plain dicts"""
        if not sources:
            return []

        result = []
        for source in sources:
            if hasattr(source, "model_dump"):  # Pydantic
                result.append(source.model_dump(mode="json"))
            elif isinstance(source, dict):
                result.append(source)
            else:  # msgspec.Struct
                result.append(msgspec.to_builtins(source))
        return result

    def to_litellm_format(self) -> dict[str, str]:
        """Convert to LiteLLM format."""
        return {"role": self.role.value, "content": self.text}


class MessageUpdateRequest(MessageRequest):
    message_id: str
    text: str  # pyright:ignore
    message_metadata: dict[str, Any] | None = None


# ============================================================================
# API OUTPUT SCHEMAS (Response DTOs)
# ============================================================================


class ConvoRequestBase(_BaseModelFlex):
    user_id: str
    workspace_id: str


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


# ============================================================================
# API OUTPUT SCHEMAS (Response DTOs)
# ============================================================================
class NewConvoResponse(ConvoRequestBase):
    title: str | None = "New Convo"
    convo_metadata: dict[str, Any] | None = None


class ConvoFullResponse(ConvoRequestBase):
    convo_id: str
    title: str | None = "New Convo"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = 0
    # Knowledge sources
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

    # Source delta
    delta: str | None = None

    # RAG sources
    rag_source: list[SourceGroupResponse] | None = None

    # Final metadata
    usage: str | None = None
    retrieval_time_ms: int | None = None
