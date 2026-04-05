"""rename convos to sessions

Revision ID: 85d62d87b8ae
Revises: 001_evolve_docs
Create Date: 2026-04-04 15:13:03.174479

"""

from typing import Sequence, Union, cast

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "85d62d87b8ae"
down_revision: Union[str, Sequence[str], None] = "001_evolve_docs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename convos to sessions and convo_id to session_id in messages."""
    # Create sessions table (copy from convos)
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column("workspace_id", sa.String(length=26), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column(
            "dynamic_prefs", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("folder_id", sa.Text(), nullable=True),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index(
        op.f("ix_sessions_session_id"), "sessions", ["session_id"], unique=False
    )
    op.create_index(op.f("ix_sessions_user_id"), "sessions", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_sessions_workspace_id"), "sessions", ["workspace_id"], unique=False
    )

    # Copy data from convos to sessions
    op.execute("""
        INSERT INTO sessions (session_id, user_id, workspace_id, title, dynamic_prefs, created_at, archived, pinned, folder_id, meta)
        SELECT convo_id, user_id, workspace_id, title, dynamic_prefs, created_at, COALESCE(archived, false), COALESCE(pinned, false), folder_id, COALESCE(meta, convo_metadata)
        FROM convos
    """)

    # Add session_id column to messages
    op.add_column(
        "messages",
        sa.Column(
            "session_id",
            sa.String(length=26),
            nullable=False,
            server_default=sa.text("''::character varying"),
        ),
    )

    # Copy data from convo_id to session_id
    op.execute("UPDATE messages SET session_id = convo_id")

    # Drop old foreign keys and indexes
    op.drop_constraint(op.f("messages_convo_id_fkey"), "messages", type_="foreignkey")
    op.drop_constraint(op.f("messages_user_id_fkey"), "messages", type_="foreignkey")
    op.drop_index(op.f("ix_messages_convo_created"), table_name="messages")

    # Create new indexes and foreign keys
    op.create_index(
        "ix_messages_session_created",
        "messages",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_foreign_key(
        None, "messages", "sessions", ["session_id"], ["session_id"], ondelete="CASCADE"
    )
    op.create_foreign_key(None, "messages", "messages", ["root_id"], ["message_id"])
    op.create_foreign_key(None, "messages", "messages", ["parent_id"], ["message_id"])

    # Drop old convo_id column
    op.drop_column("messages", "convo_id")

    # Drop old convos table and indexes
    op.drop_index(op.f("ix_convo_user_workspace"), table_name="convos")
    op.drop_index(op.f("ix_convos_convo_id"), table_name="convos")
    op.drop_table("convos")


def downgrade() -> None:
    """Revert sessions to convos."""
    # Recreate convos table
    op.create_table(
        "convos",
        sa.Column("convo_id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.String(length=26), nullable=False),
        sa.Column("workspace_id", sa.String(length=26), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column(
            "dynamic_prefs", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("folder_id", sa.Text(), nullable=True),
        sa.Column(
            "convo_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("convo_id"),
    )
    op.create_index(op.f("ix_convos_convo_id"), "convos", ["convo_id"], unique=False)
    op.create_index(
        op.f("ix_convo_user_workspace"),
        "convos",
        ["user_id", "workspace_id"],
        unique=False,
    )

    # Add convo_id back to messages
    op.add_column(
        "messages", sa.Column("convo_id", sa.String(length=26), nullable=False)
    )
    op.execute("UPDATE messages SET convo_id = session_id")

    # Drop session_id and indexes
    op.drop_index("ix_messages_session_created", table_name="messages")
    op.drop_constraint(cast(str, cast(object, None)), "messages", type_="foreignkey")
    op.drop_constraint(cast(str, cast(object, None)), "messages", type_="foreignkey")
    op.drop_constraint(cast(str, cast(object, None)), "messages", type_="foreignkey")
    op.drop_column("messages", "session_id")

    # Recreate old indexes and foreign keys
    op.create_index(
        op.f("ix_messages_convo_created"),
        "messages",
        ["convo_id", "created_at"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("messages_convo_id_fkey"),
        "messages",
        "convos",
        ["convo_id"],
        ["convo_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        op.f("messages_user_id_fkey"),
        "messages",
        "users",
        ["user_id"],
        ["user_id"],
        ondelete="SET NULL",
    )

    # Drop sessions table
    op.drop_index(op.f("ix_sessions_workspace_id"), table_name="sessions")
    op.drop_index(op.f("ix_sessions_user_id"), table_name="sessions")
    op.drop_index(op.f("ix_sessions_session_id"), table_name="sessions")
    op.drop_table("sessions")
