"""add session_state indexes for last_active_at and workspace_id

Revision ID: ebc71a2c8f4d
Revises: cd5c33c3a298
Create Date: 2026-04-07 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "ebc71a2c8f4d"
down_revision: Union[str, Sequence[str], None] = "cd5c33c3a298"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_session_states_last_active_at", "session_states", ["last_active_at"]
    )
    op.create_index(
        "ix_session_states_workspace_id", "session_states", ["workspace_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_session_states_last_active_at", "session_states")
    op.drop_index("ix_session_states_workspace_id", "session_states")
