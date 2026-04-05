"""add session_states drop old orphaned tables

Revision ID: cd5c33c3a298
Revises: 85d62d87b8ae
Create Date: 2026-04-04 15:21:03.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "cd5c33c3a298"
down_revision: Union[str, Sequence[str], None] = "85d62d87b8ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # session_states table already exists in database (populated from previous system)
    # Only drop orphaned tables that are no longer needed
    op.drop_table("chunks")
    op.drop_table("source_workspace_association")
    op.drop_table("sources")
    op.drop_table("workspaces")
    op.drop_table("users")


def downgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column("username", sa.String(length=200), nullable=True),
        sa.Column("password", sa.String(length=400), nullable=True),
        sa.Column("enabled", sa.BOOLEAN(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_login", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "preferences", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("ui_prefs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("user_id", name="users_pkey"),
    )
    op.create_table(
        "workspaces",
        sa.Column("workspace_id", sa.String(length=26), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column(
            "dynamic_prefs", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "archived_convos", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="workspaces_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("workspace_id", name="workspaces_pkey"),
        sa.UniqueConstraint(
            "user_id",
            "workspace_id",
            name="uq_user_workspace",
            postgresql_include=[],
            postgresql_nulls_not_distinct=False,
        ),
    )
    op.create_index(op.f("ix_workspaces_name"), "workspaces", ["name"], unique=False)
    op.create_table(
        "sources",
        sa.Column("source_id", sa.String(length=26), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("uploaded_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("source_summary", sa.TEXT(), nullable=False),
        sa.Column("summary_embedding", sa.TEXT(), nullable=False),
        sa.PrimaryKeyConstraint("source_id", name="sources_pkey"),
    )
    op.create_index(
        op.f("ix_sources_source_hash"), "sources", ["source_hash"], unique=True
    )
    op.create_index(op.f("ix_sources_name"), "sources", ["name"], unique=True)
    op.create_table(
        "source_workspace_association",
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("workspace_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.source_id"],
            name="source_workspace_association_source_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.workspace_id"],
            name="source_workspace_association_workspace_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "source_id", "workspace_id", name="source_workspace_association_pkey"
        ),
    )
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=26), nullable=False),
        sa.Column("source_id", sa.String(length=26), nullable=False),
        sa.Column("text", sa.TEXT(), nullable=True),
        sa.Column("start_index", sa.INTEGER(), nullable=False),
        sa.Column("end_index", sa.INTEGER(), nullable=False),
        sa.Column("token_count", sa.INTEGER(), nullable=False),
        sa.Column("chunk_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.source_id"],
            name="chunks_source_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name="chunks_pkey"),
        sa.CheckConstraint("end_index > start_index", name="check_end_after_start"),
        sa.CheckConstraint("start_index >= 0", name="check_start_index_non_negative"),
        sa.CheckConstraint("token_count > 0", name="check_token_count_positive"),
    )
    op.create_index(op.f("ix_chunks_source_id"), "chunks", ["source_id"], unique=False)
    op.drop_table("session_states")
