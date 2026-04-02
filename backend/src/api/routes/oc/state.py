"""
centralized in-memory state for the tui backend.

all mutable in-memory state lives here — session status, running tasks,
pending permission/question futures, todos, and diffs.
sub-routers import from here instead of scattering state across tui.py.

runtime state (not persisted): running_tasks, pending_permissions,
pending_questions, permission_requests, question_requests.

session state (persisted via SessionStateManager): session_status,
session_todos, session_diffs.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import loguru

if TYPE_CHECKING:
    from backend.src.api.schemas.tui_schemas import (
        FileDiff,
        PermissionRequest,
        QuestionRequest,
        Todo,
    )
    from backend.src.services.session.session_state_manager import (
        SessionStateManager,
    )

_LOG_TIME_FORMAT = os.environ.get("FOSRA_LOG_TIME_FORMAT", "full")
_TIME_FMT = "HH:mm:ss" if _LOG_TIME_FORMAT == "short" else "YYYY-MM-DD HH:mm:ss.SSS"

_state_logger = loguru.logger.bind(backend="state")
_log_path = Path.home() / ".fosra-back-state.log"
_state_logger.add(
    _log_path,
    format=f"{_TIME_FMT} | {{level: <8}} | {{name}}:{{function}}:{{line}} | {{message}}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True,
)


def _restart_marker(reason: str) -> None:
    sep = "=" * 80
    _state_logger.info(f"{sep}")
    _state_logger.info(f"NEW RUN — {reason}")
    _state_logger.info(f"{sep}")


_restart_marker("process start")


class StateDict(dict):
    """dict wrapper that logs every mutation with a live snapshot."""

    def __init__(self, name: str, logger: loguru.Logger, log_level: str = "DEBUG"):
        super().__init__()
        self._name = name
        self._logger = logger
        self._log_level = log_level

    def _log(self, op: str, key: str | None = None, value: Any = None, note: str = ""):
        snapshot = {
            k: (
                "<Task>"
                if self._name == "running_tasks" and isinstance(v, asyncio.Task)
                else v
            )
            for k, v in self.items()
        }
        extra = {"op": op, "dict": self._name}
        if key is not None:
            extra["key"] = key
        if value is not None:
            extra["value"] = str(value)[:100]
        if note:
            extra["note"] = note
        self._logger.log(
            self._log_level,
            f"[{self._name}] {op} — {json.dumps(extra)} — snapshot: {json.dumps(snapshot, default=str)[:500]}",
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._log("set", key, value)

    def __delitem__(self, key):
        self._log("del", key)
        super().__delitem__(key)

    def pop(self, key, *default):
        had_key = key in self
        if had_key or default:
            result = super().pop(key, *default)
            if had_key:
                self._log("pop", key, result)
            return result
        raise KeyError(key)

    def get(self, key, default=None):
        return super().get(key, default)

    def setdefault(self, key, default=None):
        if key not in self:
            self._log("setdefault", key, default)
            super().setdefault(key, default)
        return super().get(key, default)

    def append_to_list(self, key, value):
        if key not in self:
            super().__setitem__(key, [])
        super().__getitem__(key).append(value)
        self._log("list_append", key, value)


import json

session_status: StateDict = StateDict("session_status", _state_logger, "DEBUG")
running_tasks: StateDict = StateDict("running_tasks", _state_logger, "DEBUG")
session_todos: StateDict = StateDict("session_todos", _state_logger, "DEBUG")
session_diffs: StateDict = StateDict("session_diffs", _state_logger, "DEBUG")
pending_permissions: StateDict = StateDict(
    "pending_permissions", _state_logger, "DEBUG"
)
pending_questions: StateDict = StateDict("pending_questions", _state_logger, "DEBUG")
permission_requests: StateDict = StateDict(
    "permission_requests", _state_logger, "DEBUG"
)
question_requests: StateDict = StateDict("question_requests", _state_logger, "DEBUG")

_state_lock = asyncio.Lock()

_session_state_manager: "SessionStateManager | None" = None


async def get_session_state_manager() -> "SessionStateManager":
    global _session_state_manager
    if _session_state_manager is None:
        from backend.src.services.session.session_state_manager import (
            SessionStateManager,
        )

        _session_state_manager = await SessionStateManager.get_instance()
    return _session_state_manager


async def get_persisted_session_state(
    session_id: str,
) -> dict[str, Any] | None:
    """Get persisted session state (agent_snapshot, interaction_snapshot, etc.)."""
    manager = await get_session_state_manager()
    return await manager.get(session_id)


async def persist_session_state(
    session_id: str,
    agent_snapshot: dict[str, Any] | None = None,
    interaction_snapshot: dict[str, Any] | None = None,
    workspace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist session state to DB."""
    manager = await get_session_state_manager()
    return await manager.upsert(
        session_id=session_id,
        agent_snapshot=agent_snapshot,
        interaction_snapshot=interaction_snapshot,
        workspace_id=workspace_id,
        metadata=metadata,
    )


async def update_persisted_session_state(
    session_id: str,
    agent_snapshot: dict[str, Any] | None = None,
    interaction_snapshot: dict[str, Any] | None = None,
    workspace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Update persisted session state in DB."""
    manager = await get_session_state_manager()
    return await manager.update(
        session_id=session_id,
        agent_snapshot=agent_snapshot,
        interaction_snapshot=interaction_snapshot,
        workspace_id=workspace_id,
        metadata=metadata,
    )


def get_model_info_for_session(
    provider_id: str,
    model_id: str,
) -> dict[str, Any] | None:
    """Extract cost and limit for a model from the provider registry."""
    from backend.src.api.schemas import provider_registry

    providers = provider_registry._build_providers()
    for provider in providers:
        if provider.id == provider_id:
            model = provider.models.get(model_id)
            if model:
                return {
                    "providerID": provider_id,
                    "modelID": model_id,
                    "cost": {
                        "input": model.cost.input,
                        "output": model.cost.output,
                        "cache": {
                            "read": model.cost.cache.read,
                            "write": model.cost.cache.write,
                        },
                    },
                    "limit": {
                        "context": model.limit.context,
                        "input": model.limit.input,
                        "output": model.limit.output,
                    },
                }
    return None


async def cleanup_session(session_id: str) -> None:
    """Remove all state for a session. Call when a session is deleted."""
    _state_logger.info(
        "cleanup_session started",
        session_id=session_id,
        had_status=session_id in session_status,
        had_todos=session_id in session_todos,
        had_diffs=session_id in session_diffs,
    )

    async with _state_lock:
        session_status.pop(session_id, None)
        session_todos.pop(session_id, None)
        session_diffs.pop(session_id, None)
        permission_requests.pop(session_id, None)
        question_requests.pop(session_id, None)

    task = running_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        _state_logger.info("cleanup_session cancelled task", session_id=session_id)
    running_tasks.pop(session_id, None)

    to_remove_perm = [
        rid for rid, f in pending_permissions.items() if session_id in str(f)
    ]
    for rid in to_remove_perm:
        pending_permissions.pop(rid, None)
        _state_logger.debug(
            "cleanup_session removed pending permission future",
            session_id=session_id,
            request_id=rid,
        )

    to_remove_ques = [
        rid for rid, f in pending_questions.items() if session_id in str(f)
    ]
    for rid in to_remove_ques:
        pending_questions.pop(rid, None)
        _state_logger.debug(
            "cleanup_session removed pending question future",
            session_id=session_id,
            request_id=rid,
        )

    manager = await get_session_state_manager()
    await manager.delete(session_id)

    _state_logger.info(
        "cleanup_session completed",
        session_id=session_id,
        pending_permissions_removed=len(to_remove_perm),
        pending_questions_removed=len(to_remove_ques),
    )


def ask_permission(
    session_id: str,
    permission: str,
    patterns: list[str],
    metadata: dict[str, Any],
    always: list[str],
    tool: dict[str, Any] | None = None,
) -> tuple[str, asyncio.Future[Any], dict[str, Any]]:
    """
    register a pending permission request and return a future that blocks until the user replies.
    returns (request_id, future, request).
    the caller should await the future to get the reply ("once" | "always" | "reject").
    the request dict matches the PermissionRequest shape the TUI SDK expects.
    """
    from backend.src.storage.utils.converters import ulid_factory

    request_id = ulid_factory()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    pending_permissions[request_id] = future

    _state_logger.debug(
        "permission_future created",
        session_id=session_id,
        request_id=request_id,
        future_pending_count=len(pending_permissions),
    )

    request: dict[str, Any] = {
        "id": request_id,
        "sessionID": session_id,
        "permission": permission,
        "patterns": patterns,
        "metadata": metadata,
        "always": always,
        "tool": tool,
    }
    permission_requests.append_to_list(session_id, request)

    _state_logger.info(
        "permission_request registered",
        session_id=session_id,
        request_id=request_id,
        permission=permission,
        patterns=patterns,
    )
    return request_id, future, request


def ask_question(
    session_id: str,
    questions: list[dict[str, Any]],
    tool: dict[str, Any] | None = None,
) -> tuple[str, asyncio.Future[Any], dict[str, Any]]:
    """
    register a pending question request and return a future that blocks until the user replies.
    returns (request_id, future, request).
    the caller should await the future to get the answers list (or "reject" string).
    the request dict matches the QuestionRequest shape the TUI SDK expects.
    """
    from backend.src.storage.utils.converters import ulid_factory

    request_id = ulid_factory()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    pending_questions[request_id] = future

    _state_logger.debug(
        "question_future created",
        session_id=session_id,
        request_id=request_id,
        future_pending_count=len(pending_questions),
    )

    request: dict[str, Any] = {
        "id": request_id,
        "sessionID": session_id,
        "questions": questions,
        "tool": tool,
    }
    question_requests.append_to_list(session_id, request)

    _state_logger.info(
        "question_request registered",
        session_id=session_id,
        request_id=request_id,
        question_count=len(questions),
    )
    return request_id, future, request


def resolve_permission_future(request_id: str, result: str) -> None:
    """Resolve a pending permission future and log the outcome."""
    fut = pending_permissions.pop(request_id, None)
    if fut and not fut.done():
        fut.set_result(result)
        _state_logger.debug(
            "permission_future resolved",
            request_id=request_id,
            result=result,
        )


def resolve_question_future(
    request_id: str, result: str | list[dict[str, Any]]
) -> None:
    """Resolve a pending question future and log the outcome."""
    fut = pending_questions.pop(request_id, None)
    if fut and not fut.done():
        fut.set_result(result)
        _state_logger.debug(
            "question_future resolved",
            request_id=request_id,
            result=result,
        )
