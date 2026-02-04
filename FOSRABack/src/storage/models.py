from __future__ import annotations

from datetime import datetime, UTC
from enum import StrEnum
from typing import Any, TYPE_CHECKING, Optional
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict

from sqlalchemy import ARRAY, DDL, event
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
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)
from ulid import ULID

from FOSRABack.src.domain.enums import ConfigRole, OriginType, ToolCategory


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
    # LLM Roles
    ConfigRole.PRIMARY_LLM: ToolCategory.LLM,
    ConfigRole.FAST_LLM: ToolCategory.LLM,
    ConfigRole.HEAVY_LLM: ToolCategory.LLM,
    ConfigRole.STRATEGIC_LLM: ToolCategory.LLM,
    # Pipeline Roles
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
    tool_configs: Mapped[list["UserToolConfigORM"]] = relationship(
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

    # Metadata

    archived_convos: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )
    # Relationships
    user: Mapped["UserORM"] = relationship(back_populates="workspaces")
    convos: Mapped[list["ConvoORM"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    sources: Mapped[list["SourceORM"]] = relationship(
        secondary=source_workspace_association,
        back_populates="workspaces",
    )
    config_assignments: Mapped[list["ConfigAssignmentORM"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
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
# Config System
# ============================================================================


class UserToolConfigORM(Base):
    """
    Master library of tool configurations.

    Stores reusable configs for LLMs, embedders, parsers, storage, etc.
    Each config can be assigned to multiple workspaces/convos.
    """

    __tablename__ = "user_tool_configs"

    # Identity
    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=ulid_factory)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Metadata
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))

    # Classification
    category: Mapped[ToolCategory] = mapped_column(String(50), nullable=False)

    # Provider details
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(
        String(100)
    )  # Optional for storage, parser, etc.

    # Flexible configuration (JSON)
    # Structure varies by category:
    # - LLM: {"api_key": "...", "temperature": 0.7, "max_tokens": 1000}
    # - Storage: {"backend_type": "s3", "bucket_name": "...", "chunk_size": 8192}

    details: Mapped[Optional[dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB)
    )
    # System defaults
    is_system_default: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    # Relationships
    user: Mapped["UserORM"] = relationship(back_populates="tool_configs")
    assignments: Mapped[list["ConfigAssignmentORM"]] = relationship(
        back_populates="tool_config", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "uix_system_default_category",
            category,
            unique=True,
            postgresql_where=text("is_system_default = true"),
        ),
        Index("ix_tool_configs_user_category", "user_id", "category"),
        Index("ix_tool_configs_system_defaults", "is_system_default", "category"),
    )

    def __repr__(self) -> str:
        return (
            f"<UserToolConfig(id={self.id}, name={self.name}, "
            f"category={self.category}, provider={self.provider})>"
        )


class ConfigAssignmentORM(Base):
    """
    Assignment of configs to scopes (workspace or conversation).

    Links a UserToolConfig to a workspace or conversation,
    defining what role that config plays in that context.

    Examples:
    - Workspace 'ws-123' uses config 'gpt4-turbo' as PRIMARY_LLM
    - Conversation 'conv-456' overrides with 'claude-opus' as PRIMARY_LLM
    """

    __tablename__ = "config_assignments"

    # Identity
    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=ulid_factory)

    # Role definition
    role: Mapped[ConfigRole] = mapped_column(String(50), nullable=False)

    # Config reference
    tool_config_id: Mapped[str] = mapped_column(
        ForeignKey("user_tool_configs.id", ondelete="CASCADE"), nullable=False
    )

    # Polymorphic scope - EXACTLY ONE must be set
    workspace_id: Mapped[str | None] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE")
    )
    convo_id: Mapped[str | None] = mapped_column(
        ForeignKey("convos.convo_id", ondelete="CASCADE")  # FIXED: was "convo.convo_id"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    # Relationships
    tool_config: Mapped["UserToolConfigORM"] = relationship(
        back_populates="assignments",
        lazy="joined",  # Eager load for preference resolution
    )
    workspace: Mapped["WorkspaceORM | None"] = relationship(
        back_populates="config_assignments"
    )
    convo: Mapped["ConvoORM | None"] = relationship(back_populates="config_assignments")

    __table_args__ = (
        # No duplicate roles per workspace
        Index(
            "uix_workspace_role",
            "workspace_id",
            "role",
            unique=True,
            postgresql_where=text("workspace_id IS NOT NULL"),
        ),
        # 2. No duplicate roles per conversation (ignoring nulls)
        Index(
            "uix_convo_role",
            "convo_id",
            "role",
            unique=True,
            postgresql_where=text("convo_id IS NOT NULL"),
        ),  # Exactly one scope must be set
        CheckConstraint(
            "(workspace_id IS NOT NULL AND convo_id IS NULL) OR "
            "(workspace_id IS NULL AND convo_id IS NOT NULL)",
            name="check_exactly_one_scope",
        ),
        # Indexes for common queries
        Index("ix_assignments_workspace_role", "workspace_id", "role"),
        Index("ix_assignments_convo_role", "convo_id", "role"),
        Index("ix_assignments_config", "tool_config_id"),
    )

    @validates("role")
    def validate_role_category_match(self, key: str, role: ConfigRole) -> ConfigRole:
        """Ensure assigned tool category matches role requirements."""
        if not hasattr(self, "tool_config") or self.tool_config is None:
            return role

        expected_category = ROLE_TO_CATEGORY_MAP.get(role)
        if expected_category and self.tool_config.category != expected_category:
            raise ValueError(
                f"Cannot assign {self.tool_config.category} config "
                f"to {role} role (expected {expected_category})"
            )
        return role

    def __repr__(self) -> str:
        scope = (
            f"workspace={self.workspace_id}"
            if self.workspace_id
            else f"convo={self.convo_id}"
        )
        return f"<ConfigAssignment(role={self.role}, {scope})>"


# =============================================================================
# SQLAlchemy Events for Additional Validation
# =============================================================================


@event.listens_for(ConfigAssignmentORM, "before_insert")
@event.listens_for(ConfigAssignmentORM, "before_update")
def validate_assignment_on_save(mapper, connection, target: ConfigAssignmentORM):
    """Additional validation before insert/update."""
    expected_category = ROLE_TO_CATEGORY_MAP.get(target.role)
    if expected_category and target.tool_config.category != expected_category:
        raise ValueError(
            f"Role {target.role} requires category {expected_category}, "
            f"but config has category {target.tool_config.category}"
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
    config_assignments: Mapped[list["ConfigAssignmentORM"]] = relationship(
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
