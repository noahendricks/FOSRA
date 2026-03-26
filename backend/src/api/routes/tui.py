"""
tui-compatible rest routes + sse event stream.

all routes are mounted under /oc and return shapes the solidjs tui expects.
existing /workspaces routes are preserved for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.src.api.dependencies import (
    get_current_user_id,
    get_db_session,
    get_session_factory,
)
from backend.src.api.events import event_bus
from backend.src.api.routes.oc.state import (
    running_tasks,
    session_diffs,
    session_status,
    session_todos,
)
from backend.src.api.schemas import MessageResponse
from backend.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoUpdateRequest,
    NewConvoRequest,
)
from backend.src.api.schemas.tui_schemas import (
    DEFAULT_USER_ID,
    PROJECT_DIR,
    PromptRequest,
    convo_full_to_session,
    convo_list_item_to_session,
    convo_to_session,
    get_agents,
    get_default_config,
    get_default_provider,
    message_to_tui,
)
from backend.src.services.conversation.conversation_service import ConversationService

router = APIRouter(prefix="/oc", tags=["TUI"])

from backend.src.api.routes.oc.extended import router as extended_router
from backend.src.api.routes.oc.file import router as file_router
from backend.src.api.routes.oc.message_ops import router as message_ops_router
from backend.src.api.routes.oc.permission import router as permission_router
from backend.src.api.routes.oc.project import router as project_router
from backend.src.api.routes.oc.question import router as question_router
from backend.src.api.routes.oc.session_ops import router as session_ops_router
from backend.src.api.routes.oc.shell import router as shell_router
from backend.src.api.routes.oc.tui_events import router as tui_events_router

router.include_router(tui_events_router)
router.include_router(session_ops_router)
router.include_router(message_ops_router)
router.include_router(permission_router)
router.include_router(question_router)
router.include_router(file_router)
router.include_router(project_router)
router.include_router(shell_router)
router.include_router(extended_router)


# SSE EVENT STREAM


@router.get("/event")
async def event_stream(request: Request):
    """global sse endpoint. the tui subscribes here for all real-time events."""
    sub_id, queue = event_bus.subscribe()
    heartbeat_interval = 10

    async def generate():
        try:
            await event_bus.publish({"type": "server.connected", "properties": {}})
            last_heartbeat = 0
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    last_heartbeat = 0
                except asyncio.TimeoutError:
                    last_heartbeat += 1
                    if last_heartbeat >= heartbeat_interval:
                        yield f"data: {json.dumps({'type': 'server.heartbeat', 'properties': {}})}\n\n"
                        last_heartbeat = 0
        finally:
            event_bus.unsubscribe(sub_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        },
    )


# SESSION CRUD
@router.get("/session")
async def list_sessions(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    items = await ConversationService.list_conversations(
        session=session,
        user_id=user_id,
    )
    return [convo_list_item_to_session(item) for item in items]


@router.get("/session/status")
async def get_all_session_statuses():
    """return all tracked session statuses."""
    return session_status


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    convo = await ConversationService.get_conversation_by_id(
        session=session,
        user_id=user_id,
        convo_id=session_id,
    )
    return convo_full_to_session(convo)


@router.post("/session")
async def create_session(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    new_convo = NewConvoRequest(user_id=user_id)
    result = await ConversationService.create_conversation(
        session=session,
        new_convo=new_convo,
    )
    session_info = convo_to_session(
        convo_id=result.convo_id,
        user_id=user_id,
        title=result.title,
    )

    await event_bus.publish(
        {
            "type": "session.created",
            "properties": {"info": session_info},
        }
    )
    return session_info


@router.put("/session/{session_id}")
async def update_session(
    session_id: str,
    body: dict[str, Any],
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    update = ConvoUpdateRequest(
        user_id=user_id,
        convo_id=session_id,
        title=body.get("title"),
    )
    result = await ConversationService.update_conversation(
        session=session,
        convo_update=update,
    )
    session_info = convo_full_to_session(result)
    await event_bus.publish(
        {
            "type": "session.updated",
            "properties": {"info": session_info},
        }
    )
    return session_info


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    delete_req = ConvoDeleteRequest(user_id=user_id, convo_id=session_id)
    deleted = await ConversationService.delete_conversation(
        session=session,
        convo_request=delete_req,
    )
    if deleted:
        session_info = convo_to_session(convo_id=session_id, user_id=user_id)
        await event_bus.publish(
            {
                "type": "session.deleted",
                "properties": {"info": session_info},
            }
        )
    return deleted


# MESSAGES


@router.get("/session/{session_id}/message")
async def list_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
    limit: int | None = None,
    before: str | None = None,
):
    """
    list messages for a session, newest first.
    supports pagination via `limit` (max messages) and `before` (message ID cursor).
    """
    convo = await ConversationService.get_conversation_by_id(
        session=session,
        user_id=user_id,
        convo_id=session_id,
    )

    messages = list(convo.messages)

    if before:
        try:
            idx = next(
                i
                for i, m in enumerate(messages)
                if getattr(m, "message_id", None) == before
            )
            messages: list[MessageResponse] = messages[idx + 1 :]
        except StopIteration:
            pass

    if limit:
        messages = messages[:limit]

    result = []
    for msg in messages:
        result.append(message_to_tui(msg, session_id))
    return result


# PROMPT (fire-and-forget — events come via sse)


@router.post("/session/{session_id}/prompt")
async def prompt(
    session_id: str,
    body: PromptRequest,
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session_factory=Depends(get_session_factory),
):
    # import here to avoid circular imports
    from backend.src.services.conversation.agent_runner import run_agent_with_events

    task = asyncio.create_task(
        run_agent_with_events(
            session_id=session_id,
            user_id=user_id,
            prompt_request=body,
            session_factory=session_factory,
        )
    )
    running_tasks[session_id] = task

    # clean up when done
    def _cleanup(t):
        running_tasks.pop(session_id, None)

    task.add_done_callback(_cleanup)

    return {"ok": True}


# ABORT
@router.post("/session/{session_id}/abort")
async def abort_session(session_id: str):
    task = running_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        running_tasks.pop(session_id, None)
        session_status[session_id] = {"type": "idle"}
        await event_bus.publish(
            {
                "type": "session.status",
                "properties": {
                    "sessionID": session_id,
                    "status": {"type": "idle"},
                },
            }
        )
        return True
    return False


# TODOS


@router.get("/session/{session_id}/todo")
async def get_todos(session_id: str):
    return session_todos.get(session_id, [])


# DIFFS


@router.get("/session/{session_id}/diff")
async def get_diffs(session_id: str):
    return session_diffs.get(session_id, [])


# CONFIG


@router.get("/config")
async def get_config():
    return get_default_config()


@router.get("/config/provider")
async def get_config_providers():
    provider = get_default_provider()
    return {
        "providers": [provider],
        "default": {"litellm": "default"},
    }


# PROVIDERS


@router.get("/provider")
async def list_providers():
    provider = get_default_provider()
    return {
        "all": [provider],
        "default": {"litellm": "default"},
        "connected": ["litellm"],
    }


@router.get("/provider/auth")
async def provider_auth():
    return {}


# AGENTS


@router.get("/agent")
async def list_agents():
    return get_agents()


# COMMANDS


@router.get("/command")
async def list_commands():
    return []


# LSP / MCP / FORMATTER


@router.get("/lsp/status")
async def lsp_status():
    return []


@router.get("/mcp/status")
async def mcp_status():
    return {}


@router.get("/formatter/status")
async def formatter_status():
    return []


# VCS


@router.get("/vcs")
async def vcs_info():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"branch": branch, "dirty": False}
    except Exception:
        return {"branch": "main", "dirty": False}


# PATH


@router.get("/path")
async def get_path():
    return {
        "cwd": PROJECT_DIR,
        "root": PROJECT_DIR,
        "directory": PROJECT_DIR,
        "state": "",
        "config": "",
        "worktree": "",
    }
