from __future__ import annotations

from datetime import datetime, UTC
from enum import StrEnum
from typing import Any, TYPE_CHECKING, Optional
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict

from sqlalchemy import (
    String,
    DateTime,
    Integer,
    Boolean,
    Text,
    JSON,
    Table,
    ForeignKey,
    UniqueConstraint,
    CheckConstraint,
    Index,
    Column,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from ulid import ULID

from backend.src.domain.enums import ConfigRole, OriginType, ToolCategory


if TYPE_CHECKING:
    pass


# ============================================================================
# Base & Utilities
# ============================================================================


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


# ============================================================================
# Association Tables
# ============================================================================

source_workspace_association = Table(
    "source_workspace_association",
    Base.metadata,
    Column(
        "source_id",
        String,
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "workspace_id",
        String,
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        primary_key=True,
    ),
)
# ============================================================================
# Enums & Constants
# ============================================================================

ROLE_TO_CATEGORY_MAP: dict[ConfigRole, ToolCategory] = {
    # llM Roles
    ConfigRole.PRIMARY_LLM: ToolCategory.LLM,
    ConfigRole.FAST_LLM: ToolCategory.LLM,
    ConfigRole.HEAVY_LLM: ToolCategory.LLM,
    ConfigRole.STRATEGIC_LLM: ToolCategory.LLM,
    # pipeline Roles
    ConfigRole.DEFAULT_VECTOR_STORE: ToolCategory.VECTOR_STORE,
    ConfigRole.DEFAULT_EMBEDDER: ToolCategory.EMBEDDER,
    ConfigRole.DEFAULT_PARSER: ToolCategory.PARSER,
    ConfigRole.DEFAULT_RERANKER: ToolCategory.RERANKER,
    ConfigRole.DEFAULT_STORAGE: ToolCategory.STORAGE,
}


# ============================================================================
# User ORM
# ============================================================================


class UserORM(Base):
    """User account."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(26), default=ulid_factory, primary_key=True
    )
    username: Mapped[str | None] = mapped_column(String(200))
    password: Mapped[str | None] = mapped_column(String(length=400))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    last_login: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    workspaces: Mapped[list["WorkspaceORM"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    convos: Mapped[list["ConvoORM"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ============================================================================
# Workspace ORM
# ============================================================================


class WorkspaceORM(Base):
    """Workspace containing convos and sources."""

    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    name: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str | None] = mapped_column(String(500))

    user_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )

    archived_convos: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )
    # Relationships
    user: Mapped["UserORM"] = relationship(back_populates="workspaces")

    convos: Mapped[list["ConvoORM"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )

    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=None, nullable=True)

    sources: Mapped[list["SourceORM"]] = relationship(
        secondary=source_workspace_association,
        back_populates="workspaces",
    )

    __table_args__ = (
        UniqueConstraint("user_id", "workspace_id", name="uq_user_workspace"),
    )


# ============================================================================
# Source & Chunk ORM
# ============================================================================


class SourceORM(Base):
    """Document source."""

    __tablename__ = "sources"

    source_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    source_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    origin_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    origin_type: Mapped[OriginType] = mapped_column(String(50), nullable=False)
    unique_id: Mapped[str] = mapped_column(
        String(100), index=True, unique=True, nullable=False
    )

    # Metadata
    times_accessed: Mapped[int | None] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    source_summary: Mapped[str] = mapped_column(Text)
    summary_embedding: Mapped[str] = mapped_column(Text)

    # Relationships
    chunks: Mapped[list["ChunkORM"]] = relationship(
        back_populates="source", cascade="all, delete-orphan", lazy="selectin"
    )
    workspaces: Mapped[list["WorkspaceORM"]] = relationship(
        secondary=source_workspace_association, back_populates="sources"
    )


class ChunkORM(Base):
    """Text chunk from a source document."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    source_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False
    )

    # Content
    text: Mapped[str | None] = mapped_column(Text)

    # Position
    start_index: Mapped[int] = mapped_column(Integer, nullable=False)
    end_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    source: Mapped["SourceORM"] = relationship(back_populates="chunks")

    __table_args__ = (
        CheckConstraint("start_index >= 0", name="check_start_index_non_negative"),
        CheckConstraint("end_index > start_index", name="check_end_after_start"),
        CheckConstraint("token_count > 0", name="check_token_count_positive"),
        Index("ix_chunks_source_id", "source_id"),
    )


# ============================================================================
# Conversation & Message ORM
# ============================================================================


class ConvoORM(Base):
    """Conversation containing messages."""

    __tablename__ = "convos"

    convo_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, index=True, default=ulid_factory
    )
    user_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Metadata
    title: Mapped[str | None] = mapped_column(String(500), default="New Convo")

    convo_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )

    convo_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=None, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    user: Mapped["UserORM"] = relationship(back_populates="convos", lazy="selectin")
    workspace: Mapped["WorkspaceORM"] = relationship(
        back_populates="convos", lazy="selectin"
    )
    messages: Mapped[list["MessageORM"]] = relationship(
        back_populates="convo",  # FIXED: was "convos"
        cascade="all, delete-orphan",
    )
    __table_args__ = (Index("ix_convo_user_workspace", "user_id", "workspace_id"),)


class MessageORM(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, index=True, default=ulid_factory
    )
    convo_id: Mapped[str] = mapped_column(
        String(26),
        ForeignKey("convos.convo_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("users.user_id", ondelete="SET NULL")
    )

    # Content
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    message_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )

    attached_files: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    attached_sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    chat_metadata: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Threading support
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.message_id", ondelete="SET NULL")
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Relationships
    convo: Mapped["ConvoORM"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_messages_convo_created", "convo_id", "created_at"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="check_valid_role"
        ),
    )
