import asyncio
from datetime import UTC, datetime
from typing import Any


class SessionStateManager:
    _instance: "SessionStateManager | None" = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        self._session_factory: Any = None
        self._in_memory_state: dict[str, dict[str, Any]] = {}
        self._in_memory_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> "SessionStateManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_session_factory(self, factory: Any):
        self._session_factory = factory

    async def get(self, session_id: str) -> dict[str, Any] | None:
        async with self._in_memory_lock:
            if session_id in self._in_memory_state:
                return self._in_memory_state[session_id]

        if self._session_factory:
            async with self._session_factory() as session:
                from backend.src.storage.repos.session_state_repo import (
                    SessionStateRepo,
                )

                repo = SessionStateRepo(session)
                orm = await repo.get(session_id)
                if orm:
                    state = {
                        "session_id": orm.session_id,
                        "created_at": orm.created_at,
                        "updated_at": orm.updated_at,
                        "last_active_at": orm.last_active_at,
                        "agent_snapshot": orm.agent_snapshot,
                        "interaction_snapshot": orm.interaction_snapshot,
                        "workspace_id": orm.workspace_id,
                        "metadata": orm.metadata_,
                    }
                    async with self._in_memory_lock:
                        self._in_memory_state[session_id] = state
                    return state
        return None

    async def upsert(
        self,
        session_id: str,
        agent_snapshot: dict[str, Any] | None = None,
        interaction_snapshot: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        state = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "last_active_at": now,
            "agent_snapshot": agent_snapshot,
            "interaction_snapshot": interaction_snapshot,
            "workspace_id": workspace_id,
            "metadata": metadata,
        }

        async with self._in_memory_lock:
            existing = self._in_memory_state.get(session_id)
            if existing:
                state["created_at"] = existing.get("created_at", now)
                if agent_snapshot is not None:
                    existing["agent_snapshot"] = agent_snapshot
                if interaction_snapshot is not None:
                    existing["interaction_snapshot"] = interaction_snapshot
                if workspace_id is not None:
                    existing["workspace_id"] = workspace_id
                if metadata is not None:
                    existing["metadata"] = metadata
                existing["updated_at"] = now
                existing["last_active_at"] = now
                state = existing
            else:
                self._in_memory_state[session_id] = state

        if self._session_factory:
            async with self._session_factory() as session:
                from backend.src.storage.repos.session_state_repo import (
                    SessionStateRepo,
                )

                repo = SessionStateRepo(session)
                _ = await repo.upsert(
                    session_id=session_id,
                    agent_snapshot=agent_snapshot,
                    interaction_snapshot=interaction_snapshot,
                    workspace_id=workspace_id,
                    metadata=metadata,
                )

        return state

    async def update(
        self,
        session_id: str,
        agent_snapshot: dict[str, Any] | None = None,
        interaction_snapshot: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        async with self._in_memory_lock:
            if session_id in self._in_memory_state:
                now = datetime.now(UTC)
                state = self._in_memory_state[session_id]
                if agent_snapshot is not None:
                    state["agent_snapshot"] = agent_snapshot
                if interaction_snapshot is not None:
                    state["interaction_snapshot"] = interaction_snapshot
                if workspace_id is not None:
                    state["workspace_id"] = workspace_id
                if metadata is not None:
                    state["metadata"] = metadata
                state["updated_at"] = now
                state["last_active_at"] = now

        if self._session_factory:
            async with self._session_factory() as session:
                from backend.src.storage.repos.session_state_repo import (
                    SessionStateRepo,
                )

                repo = SessionStateRepo(session)
                _ = await repo.update(
                    session_id=session_id,
                    agent_snapshot=agent_snapshot,
                    interaction_snapshot=interaction_snapshot,
                    workspace_id=workspace_id,
                    metadata=metadata,
                )

        return await self.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._in_memory_lock:
            _ = self._in_memory_state.pop(session_id, None)

        if self._session_factory:
            async with self._session_factory() as session:
                from backend.src.storage.repos.session_state_repo import (
                    SessionStateRepo,
                )

                repo = SessionStateRepo(session)
                return await repo.delete(session_id)
        return True

    async def list_active(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._session_factory:
            async with self._session_factory() as session:
                from backend.src.storage.repos.session_state_repo import (
                    SessionStateRepo,
                )

                repo = SessionStateRepo(session)
                orm_states = await repo.list_active(limit)
            return [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "last_active_at": s.last_active_at,
                    "agent_snapshot": s.agent_snapshot,
                    "interaction_snapshot": s.interaction_snapshot,
                    "workspace_id": s.workspace_id,
                    "metadata": s.metadata_,
                }
                for s in orm_states
            ]
        return []
