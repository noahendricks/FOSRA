from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.src.storage.models import Base
from backend.src.storage.utils.converters import ulid_factory


class SessionStateORM(Base):
    __tablename__ = "session_states"

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
    agent_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    interaction_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    workspace_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
