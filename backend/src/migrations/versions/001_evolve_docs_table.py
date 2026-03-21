"""evolve docs table for file registry

Revision ID: 001_evolve_docs
Revises: e9ae3c64e926
Create Date: 2026-03-20

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001_evolve_docs"
down_revision: Union[str, Sequence[str], None] = "e9ae3c64e926"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add file registry columns to docs table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "docs" not in inspector.get_table_names():
        op.create_table(
            "docs",
            sa.Column("doc_id", sa.String(length=26), nullable=False),
            sa.Column("doc_hash", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=64), nullable=False),
            sa.Column("type", sa.String(length=50), nullable=False),
            sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("doc_summary", sa.Text(), nullable=False),
            sa.Column("summary_embedding", sa.Text(), nullable=False),
            sa.Column("path", sa.Text(), nullable=True),
            sa.Column("language", sa.Text(), nullable=True),
            sa.Column("repo", sa.Text(), nullable=True),
            sa.Column("source_type", sa.Text(), nullable=False, server_default="doc"),
            sa.Column("checksum", sa.String(length=64), nullable=True),
            sa.PrimaryKeyConstraint("doc_id"),
            sa.UniqueConstraint("path", "repo", name="uq_docs_path_repo"),
        )
        op.create_index(op.f("ix_docs_doc_hash"), "docs", ["doc_hash"], unique=True)
        op.create_index(op.f("ix_docs_name"), "docs", ["name"], unique=True)
    else:
        existing_cols = {c["name"] for c in inspector.get_columns("docs")}
        if "path" not in existing_cols:
            op.add_column("docs", sa.Column("path", sa.Text(), nullable=True))
        if "language" not in existing_cols:
            op.add_column("docs", sa.Column("language", sa.Text(), nullable=True))
        if "repo" not in existing_cols:
            op.add_column("docs", sa.Column("repo", sa.Text(), nullable=True))
        if "source_type" not in existing_cols:
            op.add_column(
                "docs",
                sa.Column(
                    "source_type", sa.Text(), nullable=False, server_default="doc"
                ),
            )
        if "checksum" not in existing_cols:
            op.add_column(
                "docs", sa.Column("checksum", sa.String(length=64), nullable=True)
            )

        conn = op.get_bind()
        constraints = inspector.get_unique_constraints("docs")
        if not any(
            c["name"] == "uq_docs_path_repo" for c in constraints if c.get("name")
        ):
            op.create_unique_constraint("uq_docs_path_repo", "docs", ["path", "repo"])


def downgrade() -> None:
    """Downgrade schema: remove file registry columns from docs table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "docs" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("docs")}

        constraints = inspector.get_unique_constraints("docs")
        if any(c["name"] == "uq_docs_path_repo" for c in constraints if c.get("name")):
            op.drop_constraint("uq_docs_path_repo", "docs", type_="unique")

        if "checksum" in existing_cols:
            op.drop_column("docs", "checksum")
        if "source_type" in existing_cols:
            op.drop_column("docs", "source_type")
        if "repo" in existing_cols:
            op.drop_column("docs", "repo")
        if "language" in existing_cols:
            op.drop_column("docs", "language")
        if "path" in existing_cols:
            op.drop_column("docs", "path")
