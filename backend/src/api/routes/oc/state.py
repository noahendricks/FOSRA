"""
centralized in-memory state for the tui backend.

all mutable in-memory state lives here — session status, running tasks,
pending permission/question futures, todos, and diffs.
sub-routers import from here instead of scattering state across tui.py.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.src.api.schemas.tui_schemas import (
        FileDiff,
        PermissionRequest,
        QuestionRequest,
        Todo,
    )

session_status: dict[str, dict[str, Any]] = {}

running_tasks: dict[str, asyncio.Task] = {}

session_todos: dict[str, list[dict[str, Any]]] = {}

session_diffs: dict[str, list[dict[str, Any]]] = {}

pending_permissions: dict[str, asyncio.Future] = {}

pending_questions: dict[str, asyncio.Future] = {}

permission_requests: dict[str, list[dict[str, Any]]] = {}

question_requests: dict[str, list[dict[str, Any]]] = {}

_state_lock = asyncio.Lock()


async def cleanup_session(session_id: str) -> None:
    """Remove all state for a session. Call when a session is deleted."""
    async with _state_lock:
        session_status.pop(session_id, None)
        session_todos.pop(session_id, None)
        session_diffs.pop(session_id, None)
        permission_requests.pop(session_id, None)
        question_requests.pop(session_id, None)

    task = running_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
    running_tasks.pop(session_id, None)

    to_remove_perm = [
        rid for rid, f in pending_permissions.items() if session_id in str(f)
    ]
    for rid in to_remove_perm:
        pending_permissions.pop(rid, None)

    to_remove_ques = [
        rid for rid, f in pending_questions.items() if session_id in str(f)
    ]
    for rid in to_remove_ques:
        pending_questions.pop(rid, None)


def ask_permission(
    session_id: str,
    permission: str,
    patterns: list[str],
    metadata: dict[str, Any],
    always: list[str],
    tool: dict[str, Any] | None = None,
) -> tuple[str, asyncio.Future]:
    """
    register a pending permission request and return a future that blocks until the user replies.
    returns (request_id, future).
    the caller should await the future to get the reply ("once" | "always" | "reject").
    """
    from backend.src.storage.utils.converters import ulid_factory

    request_id = ulid_factory()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    pending_permissions[request_id] = future

    request = {
        "id": request_id,
        "sessionID": session_id,
        "permission": permission,
        "patterns": patterns,
        "metadata": metadata,
        "always": always,
        "tool": tool,
    }
    permission_requests.setdefault(session_id, []).append(request)
    return request_id, future


def ask_question(
    session_id: str,
    questions: list[dict[str, Any]],
    tool: dict[str, Any] | None = None,
) -> tuple[str, asyncio.Future]:
    """
    register a pending question request and return a future that blocks until the user replies.
    returns (request_id, future).
    the caller should await the future to get the answers list (or "reject" string).
    """
    from backend.src.storage.utils.converters import ulid_factory

    request_id = ulid_factory()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    pending_questions[request_id] = future

    request = {
        "id": request_id,
        "sessionID": session_id,
        "questions": questions,
        "tool": tool,
    }
    question_requests.setdefault(session_id, []).append(request)
    return request_id, future
