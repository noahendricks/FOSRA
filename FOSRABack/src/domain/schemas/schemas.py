from __future__ import annotations


from msgspec import field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID


from FOSRABack.src.domain.schemas.source_schemas import (
    RetrievedResult,
    SourceFull,
    SourceGroup,
)
from FOSRABack.src.domain.enums import MessageRole
from FOSRABack.src.storage.utils.converters import (
    DomainStruct,
    utc_now,
)

import builtins

from FOSRABack.src.storage.models import ConfigRole, ToolCategory


# ============================================================================
# User Schemas
# ============================================================================
#


class User(DomainStruct):
    """Base user properties."""

    user_id: str
    username: str
    created_at: datetime | None = None
    last_login: datetime | None = None


class UserLogin(DomainStruct):
    """Base user properties."""

    user_id: str
    username: str
    password: str
    enabled: bool = True


class UserUpdate(User):
    """Properties for updating a user."""

    # TODO: Needs to actually update fields; currently doesn't upadate *only* the fields necessary
    # WARN: Incomplete / Not Working
    name: str | None = None
    enabled: bool | None = None
    updated_at: datetime = field(default_factory=utc_now)


# ============================================================================
# Workspace Schemas
# ============================================================================


class Workspace(DomainStruct):
    """Base workspace properties."""

    user_id: str
    name: str = field(default="New Workspace")
    description: str | None = field(
        default=None,
    )
    workspace_id: str | None = None
    archived_convos: list[str] | None = None


class WorkspaceUpdateRequest(Workspace):
    """Properties for updating a workspace."""

    name: str | None = field(
        default=None,
    )
    description: str | None = field(
        default=None,
    )


class WorkspaceFull(Workspace):
    """Workspace with related entities loaded."""

    sources_count: int = 0
    conversations_count: int = 0


# ============================================================================
# Message Schemas
# ============================================================================


class FilePartDomain(DomainStruct):
    """A File part of a UI message (matches AI SDK format)"""

    type: str
    name: str
    size: int
    filename: str
    bytes: builtins.bytes | None
    media_type: str
    url: str | None = None


class Message(DomainStruct):
    """Message for User and Assistant"""

    role: MessageRole
    convo_id: str
    text: str
    user_id: str | None = None
    message_id: str | None = None
    metadata: None = None
    attached_files: list[FilePartDomain] | None = None
    attached_sources: list[dict[str, Any]] | None = None
    sources_count: int = 0
    timestamp: datetime = field(default_factory=lambda: utc_now())
    message_metadata: dict[str, Any] | None = None

    def to_litellm_format(self) -> dict[str, str]:
        """Convert to LiteLLM format."""
        return {"role": self.role.value, "content": self.text}


class MessageUpdate(Message, kw_only=True):
    text: str | None = None


# ============================================================================
# Conversation Schemas
# ============================================================================


class Convo(DomainStruct):
    user_id: str
    convo_id: str
    workspace_id: str
    title: str | None = "New Convo"


class NewConvo(Convo, kw_only=True):
    convo_id: str = field(default_factory=lambda: str(UUID()))
    convo_metadata: dict[str, Any] | None = None


class ConvoUpdate(Convo):
    convo_metadata: dict[str, Any] | None = None
    messages: list[Message] | None = None
    data: dict[str, Any] | None = None


class ConvoFull(Convo):
    created_at: datetime = field(default_factory=utc_now)
    knowledge_sources: list[SourceFull] = field(
        default_factory=list,
    )
    messages: list[Message] = field(default_factory=list)


class Completion(DomainStruct):
    message: Message
    usage: str = field(default="")
    retrieval_time_ms: int | None = field(
        default=None,
    )

    finish_reason: str | None = field(
        default=None,
    )


class StreamChunk(DomainStruct):
    """Individual chunk in a streaming response."""

    type: Literal["content", "rag_sources", "done"]

    # Source delta
    delta: str | None = None

    # RAG sources
    rag_sources: list[SourceFull] | None = None

    # Final metadata
    usage: str | None = None
    retrieval_time_ms: int | None = None


class ValidationResult(DomainStruct):
    """Result of LLM configuration validation."""

    is_valid: bool
    error_message: str = ""
    response_preview: str | None = None


class ToolConfig(DomainStruct):
    # Config details
    name: str
    category: ToolCategory
    provider: str
    model: str | None = None
    details: dict[str, Any] = {}
    description: str = ""
    is_system_default: bool | None = True

    # Optional immediate assignment
    assign_to_workspace: str | None = None
    assign_to_convo: str | None = None
    assign_as_role: ConfigRole | None = None

    def should_assign(self) -> bool:
        return self.assign_as_role is not None


class ToolConfigUpdate(DomainStruct):
    name: str | None = None
    description: str | None = None
    provider: str | None = None
    model: str | None = None
    details: dict[str, Any] | None = None


class ToolConfigResponse(DomainStruct):
    id: str
    user_id: str
    name: str
    description: str | None
    category: ToolCategory
    provider: str
    model: str | None
    details: dict[str, Any]
    created_at: str
    updated_at: str
    is_system_default: bool = True
