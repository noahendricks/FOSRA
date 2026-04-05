from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.storage.models import SessionStateORM


class SessionStateRepo:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, session_id: str) -> SessionStateORM | None:
        result = await self._session.execute(
            select(SessionStateORM).where(SessionStateORM.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        session_id: str,
        agent_snapshot: dict[str, Any] | None = None,
        interaction_snapshot: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionStateORM:
        now = datetime.now(UTC)
        existing = await self.get(session_id)

        if existing:
            existing.updated_at = now
            existing.last_active_at = now
            if agent_snapshot is not None:
                existing.agent_snapshot = agent_snapshot
            if interaction_snapshot is not None:
                existing.interaction_snapshot = interaction_snapshot
            if workspace_id is not None:
                existing.workspace_id = workspace_id
            if metadata is not None:
                existing.metadata_ = metadata
            await self._session.commit()
            await self._session.refresh(existing)
            return existing
        else:
            new_state = SessionStateORM(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                last_active_at=now,
                agent_snapshot=agent_snapshot,
                interaction_snapshot=interaction_snapshot,
                workspace_id=workspace_id,
                metadata_=metadata,
            )
            self._session.add(new_state)
            await self._session.commit()
            await self._session.refresh(new_state)
            return new_state

    async def update(
        self,
        session_id: str,
        agent_snapshot: dict[str, Any] | None = None,
        interaction_snapshot: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionStateORM | None:
        now = datetime.now(UTC)
        updates: dict[str, Any] = {"updated_at": now, "last_active_at": now}
        if agent_snapshot is not None:
            updates["agent_snapshot"] = agent_snapshot
        if interaction_snapshot is not None:
            updates["interaction_snapshot"] = interaction_snapshot
        if workspace_id is not None:
            updates["workspace_id"] = workspace_id
        if metadata is not None:
            updates["metadata_"] = metadata

        _ = await self._session.execute(
            update(SessionStateORM)
            .where(SessionStateORM.session_id == session_id)
            .values(**updates)
        )
        await self._session.commit()
        return await self.get(session_id)

    async def delete(self, session_id: str) -> bool:
        result = await self._session.execute(
            delete(SessionStateORM).where(SessionStateORM.session_id == session_id)
        )
        await self._session.commit()
        await self._session.refresh(existing := await self.get(session_id))
        rc = getattr(result, "rowcount", 0) or 0
        return rc > 0

    async def list_active(self, limit: int = 50) -> list[SessionStateORM]:
        result = await self._session.execute(
            select(SessionStateORM)
            .order_by(SessionStateORM.last_active_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
