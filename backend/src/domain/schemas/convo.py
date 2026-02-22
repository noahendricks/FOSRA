import builtins
from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from msgspec import field
from backend.src.domain.enums import MessageRole
from backend.src.domain.schemas.doc import Doc, MDNFile
from backend.src.storage.models import utc_now
from backend.src.storage.utils.converters import DomainStruct


# vercel / browser returned file


class Message(DomainStruct):
    role: MessageRole
    convo_id: str
    text: str
    user_id: str | None = None
    message_id: str | None = None
    parent_id: str | None = None
    root_id: str | None = None
    metadata: None = None
    attached_files: list[MDNFile] | None = None
    attached_sources: list[dict[str, Any]] | None = None
    sources_count: int = 0
    timestamp: datetime = field(default_factory=lambda: utc_now())
    message_metadata: dict[str, Any] | None = None

    def to_litellm_format(self) -> dict[str, str]:
        """Convert to LiteLLM format."""
        return {"role": self.role.value, "content": self.text}


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
    docs: list[Doc] = []
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

    docs: list[Doc] | None = None

    # Final metadata
    usage: str | None = None
    retrieval_time_ms: int | None = None
