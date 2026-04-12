"""add messages_v2 and parts tables

Revision ID: 7f3e2a1b8c4d
Revises: ebc71a2c8f4d
Create Date: 2026-04-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "7f3e2a1b8c4d"
down_revision: Union[str, Sequence[str], None] = "ebc71a2c8f4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "messages_v2",
        sa.Column("message_id", sa.String(length=26), nullable=False),
        sa.Column("session_id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.String(length=26), nullable=True),
        sa.Column("parent_id", sa.String(length=26), nullable=True),
        sa.Column("root_id", sa.String(length=26), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("text", sa.TEXT(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "message_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name="messages_v2_session_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["messages_v2.message_id"],
            name="messages_v2_parent_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["root_id"],
            ["messages_v2.message_id"],
            name="messages_v2_root_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", name="messages_v2_pkey"),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="check_valid_role_v2",
        ),
    )
    op.create_index(
        "ix_messages_v2_session_created", "messages_v2", ["session_id", "created_at"]
    )

    op.create_table(
        "parts",
        sa.Column("part_id", sa.String(length=26), nullable=False),
        sa.Column("message_id", sa.String(length=26), nullable=False),
        sa.Column("session_id", sa.String(length=26), nullable=False),
        sa.Column("part_type", sa.String(length=50), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages_v2.message_id"],
            name="parts_message_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name="parts_session_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("part_id", name="parts_pkey"),
    )
    op.create_index("ix_parts_message_id", "parts", ["message_id"])
    op.create_index("ix_parts_session_id", "parts", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_parts_session_id", "parts")
    op.drop_index("ix_parts_message_id", "parts")
    op.drop_table("parts")
    op.drop_index("ix_messages_v2_session_created", "messages_v2")
    op.drop_table("messages_v2")
