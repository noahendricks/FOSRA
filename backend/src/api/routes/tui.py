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
from typing import Annotated, Any, AsyncIterable

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from loguru import logger

from backend.src.api.dependencies import (
    get_current_user_id,
    get_db_session,
    get_session_factory,
)
from backend.src.api.events import event_bus
from backend.src.api.routes.oc.state import (
    cleanup_session,
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
from backend.src.api.schemas.provider_registry import (
    get_config_providers_response,
    get_provider_list_response,
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
    message_to_tui,
)
from backend.src.services.conversation.conversation_service import ConversationService

router = APIRouter(prefix="/oc", tags=["TUI"])

from backend.src.api.routes.oc.file import router as file_router
from backend.src.api.routes.oc.message_ops import router as message_ops_router
from backend.src.api.routes.oc.permission import router as permission_router
from backend.src.api.routes.oc.project import router as project_router
from backend.src.api.routes.oc.question import router as question_router
from backend.src.api.routes.oc.session_ops import router as session_ops_router
from backend.src.api.routes.oc.shell import router as shell_router
from backend.src.api.routes.oc.skill import router as skill_router

router.include_router(session_ops_router)
router.include_router(message_ops_router)
router.include_router(permission_router)
router.include_router(question_router)
router.include_router(file_router)
router.include_router(project_router)
router.include_router(shell_router)
router.include_router(skill_router)


@router.get("/event", response_class=EventSourceResponse)
async def sse_endpoint(
    request: Request,
    last_event_id: int | None = Header(None, alias="Last-Event-ID"),
) -> AsyncIterable[ServerSentEvent]:
    """
    Global SSE endpoint. The TUI subscribes here for all real-time events.

    FastAPI SSE provides:
    - 15s keep-alive pings automatically
    - Cache-Control: no-cache, X-Accel-Buffering: no
    - Last-Event-ID reconnection replay via replay_missed()
    """
    sub_id, queue = event_bus.subscribe()

    async def event_generator():
        try:
            for missed in event_bus.replay_missed(last_event_id or 0):
                yield ServerSentEvent(
                    data=json.dumps(
                        {"type": missed.type, "properties": missed.properties}
                    ),
                    event=missed.type,
                    id=str(missed.sequence_nr),
                    retry=5000,
                )

            yield ServerSentEvent(
                data=json.dumps({"type": "server.connected", "properties": {}}),
                event="server.connected",
                id="0",
                retry=5000,
            )

            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield ServerSentEvent(
                        data=json.dumps({"type": ev.type, "properties": ev.properties}),
                        event=ev.type,
                        id=str(ev.sequence_nr),
                        retry=5000,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            event_bus.unsubscribe(sub_id)

    return event_generator()


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
            "properties": {"info": session_info.model_dump()},
        }
    )
    return session_info


@router.patch("/session/{session_id}")
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
            "properties": {"info": session_info.model_dump()},
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
        await cleanup_session(session_id)
        session_info = convo_to_session(convo_id=session_id, user_id=user_id)
        await event_bus.publish(
            {
                "type": "session.deleted",
                "properties": {"info": session_info.model_dump()},
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
@router.get("/config/providers")
async def get_config_providers():
    return get_config_providers_response()


# PROVIDERS


@router.get("/provider")
async def list_providers():
    return get_provider_list_response()


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


# Path aliases (canonical paths without /status suffix)
@router.get("/lsp")
async def lsp_alias():
    return []


@router.get("/mcp")
async def mcp_alias():
    return {}


@router.get("/formatter")
async def formatter_alias():
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


# VERSION

SERVER_VERSION = "0.2.0"


@router.get("/version")
async def get_version():
    return {"version": SERVER_VERSION}
