from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from ulid import ULID

from backend.src.domain.enums import ConfigRole, SourceType, ToolCategory
from backend.src.domain.schemas.config import UserPreferences

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
    # llm roles
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


class DocORM(Base):
    """Document source."""

    __tablename__ = "sources"

    doc_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )

    doc_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    type: Mapped[SourceType] = mapped_column(String(50), nullable=False)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # CHUNK VECTOR ID'S

    doc_summary: Mapped[str] = mapped_column(Text)

    summary_embedding: Mapped[str] = mapped_column(Text)


class DocTopicORM(Base):
    """Document Topics ."""

    __tablename__ = "sources"

    topic_id: Mapped[str] = mapped_column(
        String(26), primary_key=True, default=ulid_factory
    )
    topic: Mapped[str] = mapped_column(String(length=200))

    topic_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    icl_examples: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
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

    title: Mapped[str | None] = mapped_column(String(500), default="New Convo")

    dynamic_prefs: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    archived: Mapped[Boolean] = mapped_column(Boolean, default=False)

    pinned: Mapped[Boolean] = mapped_column(Boolean, default=False)

    messages: Mapped[list["MessageORM"]] = relationship(
        back_populates="convo",
        cascade="all, delete-orphan",
    )

    folder_id = mapped_column(Text, nullable=True)

    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )


class MessageORM(Base):
    """Individual message in a conversation."""

    __tablename__: str = "messages"

    # ids
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

    # id of message right above
    parent_id: Mapped[str | None] = mapped_column(String)

    # [Message 1] <- Root [M1 v2]<- child [M1 v3]
    root_id: Mapped[str | None] = mapped_column(String)

    root_message = relationship(
        "MessageORM", back_populates="child_messages", remote_side=[message_id]
    )

    child_messages = relationship("MessageORM", back_populates="root_message")

    role: Mapped[str] = mapped_column(String(20), nullable=False)

    # content
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # sources / files
    attached_files: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    attached_sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    # time
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    message_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )

    # relationships
    convo: Mapped["ConvoORM"] = relationship(back_populates="messages")

    # args
    __table_args__ = (
        Index("ix_messages_convo_created", "convo_id", "created_at"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="check_valid_role"
        ),
    )
