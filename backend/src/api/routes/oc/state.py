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
