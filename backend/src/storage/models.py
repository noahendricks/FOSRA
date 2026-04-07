from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from ulid import ULID

from backend.src.domain.enums import (
    ConfigRole,
    FileSourceType,
    SourceType,
    ToolCategory,
)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
    }


def ulid_factory() -> str:
    return str(ULID())


def utc_now() -> datetime:
    return datetime.now(UTC)


ROLE_TO_CATEGORY_MAP: dict[ConfigRole, ToolCategory] = {
    ConfigRole.PRIMARY_LLM: ToolCategory.LLM,
    ConfigRole.FAST_LLM: ToolCategory.LLM,
    ConfigRole.HEAVY_LLM: ToolCategory.LLM,
    ConfigRole.STRATEGIC_LLM: ToolCategory.LLM,
    ConfigRole.DEFAULT_VECTOR_STORE: ToolCategory.VECTOR_STORE,
    ConfigRole.DEFAULT_EMBEDDER: ToolCategory.EMBEDDER,
    ConfigRole.DEFAULT_PARSER: ToolCategory.PARSER,
    ConfigRole.DEFAULT_RERANKER: ToolCategory.RERANKER,
    ConfigRole.DEFAULT_STORAGE: ToolCategory.STORAGE,
}


class DocORM(Base):
    __tablename__ = "docs"

    doc_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )

    doc_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(64), index=True)

    type: Mapped[SourceType] = mapped_column(String(50), nullable=False)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(
        Text, nullable=False, default=FileSourceType.DOC
    )
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (UniqueConstraint("path", "repo", name="uq_docs_path_repo"),)

    doc_summary: Mapped[str] = mapped_column(Text)

    summary_embedding: Mapped[str] = mapped_column(Text)


class DocTopicORM(Base):
    __tablename__ = "doc_topics"

    topic_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    topic: Mapped[str] = mapped_column(String(length=200))

    topic_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    icl_examples: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )


class SessionORM(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, index=True, default=ulid_factory
    )

    user_id: Mapped[str] = mapped_column(String(26), nullable=False, index=True)

    workspace_id: Mapped[str] = mapped_column(String(26), nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(String(500), default="New Session")

    directory: Mapped[str] = mapped_column(Text, nullable=False)

    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1")

    parent_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("sessions.session_id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    archived: Mapped[Boolean] = mapped_column(Boolean, default=False)
    pinned: Mapped[Boolean] = mapped_column(Boolean, default=False)

    messages: Mapped[list["MessageORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    folder_id = mapped_column(Text, nullable=True)

    dynamic_prefs: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )
    meta: Mapped[dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSONB))

    permission: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    revert: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class MessageORM(Base):
    __tablename__: str = "messages"

    message_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, index=True, default=ulid_factory
    )

    session_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id: Mapped[str | None] = mapped_column(String(26), nullable=True)

    parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("messages.message_id"), nullable=True
    )

    root_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("messages.message_id"), nullable=True
    )

    root_message = relationship(
        "MessageORM",
        back_populates="child_messages",
        remote_side=[message_id],
        foreign_keys=[root_id],
    )

    child_messages = relationship(
        "MessageORM",
        back_populates="root_message",
        foreign_keys=[root_id],
    )

    role: Mapped[str] = mapped_column(String(20), nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    attached_files: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    attached_sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    message_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )

    session: Mapped["SessionORM"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="check_valid_role"
        ),
    )


class SessionStateORM(Base):
    __tablename__ = "session_states"
    __table_args__ = (
        Index("ix_session_states_last_active_at", "last_active_at"),
        Index("ix_session_states_workspace_id", "workspace_id"),
    )

    session_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.now(UTC),
        onupdate=datetime.now(UTC),
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    agent_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    interaction_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    workspace_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
