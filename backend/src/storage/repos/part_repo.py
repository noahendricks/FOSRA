from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.storage.models import PartORM


class PartRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, part_id: str) -> PartORM | None:
        result = await self._session.execute(
            select(PartORM).where(PartORM.part_id == part_id)
        )
        return result.scalar_one_or_none()

    async def get_by_message(self, message_id: str) -> list[PartORM]:
        result = await self._session.execute(
            select(PartORM)
            .where(PartORM.message_id == message_id)
            .order_by(PartORM.created_at)
        )
        return list(result.scalars().all())

    async def upsert(
        self,
        part_id: str,
        message_id: str,
        session_id: str,
        part_type: str,
        data: dict[str, Any],
    ) -> PartORM:
        now = datetime.now(UTC)
        existing = await self.get(part_id)

        if existing:
            existing.part_type = part_type
            existing.data = data
            existing.updated_at = now
            await self._session.commit()
            await self._session.refresh(existing)
            return existing
        else:
            new_part = PartORM(
                part_id=part_id,
                message_id=message_id,
                session_id=session_id,
                part_type=part_type,
                data=data,
                created_at=now,
                updated_at=now,
            )
            self._session.add(new_part)
            await self._session.commit()
            await self._session.refresh(new_part)
            return new_part

    async def update_data(self, part_id: str, data: dict[str, Any]) -> PartORM | None:
        now = datetime.now(UTC)
        part = await self.get(part_id)
        if not part:
            return None
        part.data = data
        part.updated_at = now
        await self._session.commit()
        await self._session.refresh(part)
        return part

    async def delete(self, part_id: str) -> bool:
        result = await self._session.execute(
            delete(PartORM).where(PartORM.part_id == part_id)
        )
        await self._session.commit()
        rc = getattr(result, "rowcount", 0) or 0
        return rc > 0

    async def delete_by_message(self, message_id: str) -> int:
        result = await self._session.execute(
            delete(PartORM).where(PartORM.message_id == message_id)
        )
        await self._session.commit()
        rc = getattr(result, "rowcount", 0) or 0
        return rc
