from datetime import datetime
from typing import Any, Literal, override
from uuid import UUID

from msgspec import field

from backend.src.domain.enums import MessageRole
from backend.src.domain.schemas.doc import Doc, MDNFile
from backend.src.storage.models import utc_now
from backend.src.storage.utils.converters import DomainStruct


class SessionTime(DomainStruct):
    created: int | datetime
    updated: int | datetime
    compacting: int | None = None
    archived: int | None = None


class SessionRevert(DomainStruct):
    message_id: str
    part_id: str | None = None
    snapshot: str | None = None
    diff: str | None = None


class SessionMetadata(DomainStruct):
    model: dict[str, Any] | None = None
    agent: str | None = None


class Message(DomainStruct):
    role: MessageRole
    session_id: str
    text: str
    user_id: str | None = None
    message_id: str | None = None
    parent_id: str | None = None
    root_id: str | None = None
    metadata: dict[str, Any] | None = None
    attached_files: list[MDNFile] | None = None
    attached_sources: list[dict[str, Any]] | None = None
    sources_count: int = 0
    timestamp: datetime = field(default_factory=lambda: utc_now())
    message_metadata: dict[str, Any] | None = None

    def to_litellm_format(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.text}


class SessionBare(DomainStruct):
    user_id: str
    session_id: str
    title: str


class Session(SessionBare, kw_only=True):
    directory: str
    version: str
    time: SessionTime
    title: str = "New Session"
    parent_id: str | None = None
    permission: dict[str, Any] | None = None
    revert: SessionRevert | None = None
    metadata: SessionMetadata | None = None


class NewSession(Session, kw_only=True):
    session_id: str = field(default_factory=lambda: str(UUID()))
    session_metadata: dict[str, Any] | None = None


class SessionUpdate(Session):
    session_metadata: dict[str, Any] | None = None
    messages: list[Message] | None = None
    data: dict[str, Any] | None = None


class SessionFull(Session):
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
    type: Literal["content", "rag_sources", "done"]
    delta: str | None = None
    docs: list[Doc] | None = None
    usage: str | None = None
    retrieval_time_ms: int | None = None
