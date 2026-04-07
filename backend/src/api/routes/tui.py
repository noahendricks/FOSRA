"""
tui-compatible rest routes + sse event stream.

all routes are mounted under /oc and return shapes the solidjs tui expects.
existing /workspaces routes are preserved for backward compatibility.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from typing import Annotated, Any, cast

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
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import (
    get_current_user_id,
    get_db_session,
    get_session_factory,
)
from backend.src.api.routes.oc.state import (
    check_session_existing,
    cleanup_session,
    get_persisted_session_state,
    pending_permissions,
    pending_questions,
    permission_requests,
    persist_session_state,
    question_requests,
    running_tasks,
    session_diffs,
    session_status,
)
from backend.src.api.schemas.provider_registry import (
    get_config_providers_response,
    get_provider_list_response,
)
from backend.src.api.schemas.session_api_schemas import (
    NewSessionRequest,
    SessionDeleteRequest,
    SessionUpdateRequest,
)
from backend.src.api.schemas.tui_schemas import (
    PROJECT_DIR,
    PromptRequest,
    Session,
    SessionTime,
    get_agents,
    get_default_config,
    message_to_tui,
    session_full_to_session,
    session_list_item_to_session,
)
from backend.src.services.session.conversation_service import SessionService
from backend.src.services.session.event_emitter import get_event_emitter

router = APIRouter(prefix="/oc", tags=["TUI"])
event_emitter = get_event_emitter()

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
    client_version: str | None = Header(None, alias="X-Client-Version"),
) -> AsyncIterable[ServerSentEvent]:
    """global sse endpoint. the tui subscribes here for all real-time events."""
    sub_id, queue = event_emitter.subscribe()

    if client_version and client_version != SERVER_VERSION:
        await event_emitter.emit_installation_update_available(
            client_version, SERVER_VERSION
        )

    try:
        for missed in event_emitter.replay_missed(last_event_id or 0):
            yield ServerSentEvent(
                data={"type": missed.type, "properties": missed.properties},
                event=missed.type,
                id=str(missed.sequence_nr),
                retry=5000,
            )

        yield ServerSentEvent(
            data={"type": "server.connected", "properties": {}},
            event="server.connected",
            id="0",
            retry=5000,
        )

        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield ServerSentEvent(
                    data={"type": ev.type, "properties": ev.properties},
                    event=ev.type,
                    id=str(ev.sequence_nr),
                    retry=5000,
                )
            except asyncio.TimeoutError:
                yield ServerSentEvent(comment="keepalive")
            except asyncio.CancelledError:
                break
    finally:
        event_emitter.unsubscribe(sub_id)


# SESSION CRUD
@router.get("/session")
async def list_sessions(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),
):
    items = await SessionService.list_sessions(
        session=session,
        user_id=user_id,
    )
    results = [session_list_item_to_session(item) for item in items]

    for item, result in zip(items, results):
        persisted = await get_persisted_session_state(item.session_id)
        if persisted and persisted.get("metadata"):
            from backend.src.api.schemas.session_schemas import SessionMetadataModel

            result.metadata = SessionMetadataModel(**persisted["metadata"])

    return results


@router.get("/session/status")
async def get_all_session_statuses():
    """return all tracked session statuses."""
    return session_status


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
):
    session_obj = await SessionService.get_session_by_id(
        session=session,
        user_id=user_id,
        session_id=session_id,
    )
    result = session_full_to_session(session_obj)

    persisted = await get_persisted_session_state(session_id)
    if persisted and persisted.get("metadata"):  # type: ignore[reportUnknownMemberType]
        from backend.src.api.schemas.session_schemas import SessionMetadataModel

        result.metadata = SessionMetadataModel(**persisted["metadata"])  # type: ignore[reportUnknownMemberType]

    return result


@router.post("/session")
async def create_session(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
):
    new_session = NewSessionRequest(user_id=user_id)
    result = await SessionService.create_session(
        session=session,
        new_session=new_session,
    )
    now = int(datetime.now(timezone.utc).timestamp())
    session_info = Session(
        id=result.session_id,
        directory=PROJECT_DIR,
        title=result.title or "New Session",
        version="",
        time=SessionTime(created=now, updated=now),
    )

    await event_emitter.emit_session_created(session_info.model_dump())  # type: ignore[reportUnknownMemberType]
    return session_info


@router.patch("/session/{session_id}")
async def update_session(
    session_id: str,
    body: dict[str, Any],  # type: ignore[reportExplicitAny]
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
):
    update = SessionUpdateRequest(
        user_id=user_id,
        session_id=session_id,
        title=body.get("title"),  # type: ignore[reportAny]
    )
    result = await SessionService.update_session(
        session=session,
        session_update=update,
    )
    session_info = session_full_to_session(result)
    await event_emitter.emit_session_updated(session_info.model_dump())  # type: ignore[reportUnknownMemberType]
    return session_info


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
):
    delete_req = SessionDeleteRequest(user_id=user_id, session_id=session_id)
    deleted = await SessionService.delete_session(
        session=session,
        session_request=delete_req,
    )
    if deleted:
        await cleanup_session(session_id)
        session_info = Session(
            id=session_id,
            directory=PROJECT_DIR,
            title="",
            version="",
            time=SessionTime(created=0, updated=0),
        )
        await event_emitter.emit_session_deleted(session_info.model_dump())  # type: ignore[reportUnknownMemberType]
    return deleted


# MESSAGES


@router.get("/session/{session_id}/message")
async def list_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
    limit: int | None = None,
    before: str | None = None,
):
    """
    list messages for a session, newest first.
    supports pagination via `limit` (max messages) and `before` (message ID cursor).
    """
    session_obj = await SessionService.get_session_by_id(
        session=session,
        user_id=user_id,
        session_id=session_id,
    )

    messages = list(session_obj.messages)

    if before:
        try:
            idx = next(
                i
                for i, m in enumerate(messages)
                if getattr(m, "message_id", None) == before
            )
            messages = messages[idx + 1 :]
        except StopIteration:
            pass

    if limit:
        messages = messages[:limit]

    result = []
    for msg in messages:
        result.append(message_to_tui(msg, session_id))
    return result


# CREATE MESSAGE (fire-and-forget — events come via sse)
@router.post("/session/{session_id}/message")
async def create_message(
    session_id: str,
    body: PromptRequest,
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session_factory=Depends(get_session_factory),  # type: ignore[reportExplicitAny]
):
    # import here to avoid circular imports
    from backend.src.services.session.agent_runner import run_agent_with_events

    provider_id = (body.model.providerID if body.model else None) or body.providerID
    model_id = (body.model.modelID if body.model else None) or body.modelID

    logger.bind(
        _structured={"prompt request body": body, "session_id": session_id}
    ).debug("[PROMPT REQUEST BODY]")

    # store model info in session metadata synchronously
    existing = await check_session_existing(session_id=session_id)
    if not existing:
        existing = await persist_session_state(
            session_id=session_id,
            metadata={"model": {"providerID": provider_id, "modelID": model_id}},
        )

        logger.bind(_structured={"persist session state": existing}).debug(
            "[PERSIST SESSION STATE]"
        )
    else:
        existing = True
    if provider_id and model_id:
        from backend.src.api.routes.oc.state import (
            get_model_info_for_session,
            update_persisted_session_state,
        )

        metadata_update: dict[str, Any] = {}  # type: ignore[reportExplicitAny]
        model_info = get_model_info_for_session(provider_id, model_id)

        if existing:
            if model_info:
                metadata_update["model"] = model_info
            if body.agent:
                metadata_update["agent"] = body.agent

            if metadata_update:
                _ = await update_persisted_session_state(
                    session_id=session_id,
                    metadata=metadata_update,
                )
                # emit session.updated so TUI refreshes session with new metadata
                updated_session = await get_persisted_session_state(session_id)
                if updated_session:
                    from backend.src.api.schemas.session_schemas import (
                        SessionMetadataModel,
                    )

                    session_data: dict[str, Any] = {"id": session_id}  # type: ignore[reportExplicitAny]
                    if updated_session.get("metadata"):  # type: ignore[reportUnknownMemberType]
                        session_data["metadata"] = SessionMetadataModel(
                            model=updated_session["metadata"].get("model"),  # type: ignore[reportUnknownMemberType]
                            agent=updated_session["metadata"].get("agent"),  # type: ignore[reportUnknownMemberType]
                        ).model_dump()

                    logger.bind(
                        _structured={"emitted session data": session_data}
                    ).debug("[EMITTED SESSION DATA]")

                    await event_emitter.emit_session_updated(session_data)

    async def _run():
        try:
            async with asyncio.timeout(300):
                await run_agent_with_events(
                    session_id=session_id,
                    user_id=user_id,
                    prompt_request=body,
                    session_factory=session_factory,
                )
        except asyncio.TimeoutError:
            await event_emitter.emit_session_error(
                session_id, "Agent execution timed out after 300 seconds"
            )
        except Exception as exc:
            await event_emitter.emit_session_error(session_id, str(exc))

    # could be taskiq instead
    task = asyncio.create_task(_run())
    running_tasks[session_id] = task

    # clean up when done
    def _cleanup(t: Any) -> None:  # type: ignore[reportExplicitAny]
        running_tasks.pop(session_id, None)

    task.add_done_callback(_cleanup)

    return {"ok": True}


# ABORT
@router.post("/session/{session_id}/abort")
async def abort_session(session_id: str):
    logger.info(
        "abort_session started",
        session_id=session_id,
        task_found=session_id in running_tasks,
    )

    task = running_tasks.get(session_id)

    if task and not task.done():
        permission_reqs = permission_requests.pop(session_id, [])
        question_reqs = question_requests.pop(session_id, [])

        for req in permission_reqs:
            fut = pending_permissions.pop(req["id"], None)
            if fut and not fut.done():
                fut.set_result("reject")

        for req in question_reqs:
            fut = pending_questions.pop(req["id"], None)
            if fut and not fut.done():
                fut.set_result("reject")

        task.cancel()
        running_tasks.pop(session_id, None)
        session_status[session_id] = {"type": "idle"}
        await event_emitter.emit_session_status(session_id, {"type": "idle"})

        logger.info(
            "abort_session completed",
            session_id=session_id,
            permission_requests_resolved=len(permission_reqs),
            question_requests_resolved=len(question_reqs),
        )
        return True

    logger.info("abort_session no active task", session_id=session_id)
    return False


# TODOS


# @router.get("/session/{session_id}/todo")
# async def get_todos(session_id: str):
#     return session_todos.get(session_id, [])


# @router.post("/session/{session_id}/todo")
# async def create_todo(
#     session_id: str,
#     body: dict[str, Any],
#     user_id: Annotated[str, Depends(get_current_user_id)],
#     session=Depends(get_db_session),
# ):
#     """create a todo for a session."""
#     from backend.src.api.schemas.tui_control_schemas import Todo

#     # check if the session exists
#     if not await session_exists(session_id, session):
#         return {"error": "session not found"}, 404

#     # check if the user is authorized to create a todo for this session
#     if not await is_user_authorized(session_id, user_id, session):
#         return {"error": "user not authorized"}, 403

#     # create the todo and add it to the session todos list
#     todo = Todo(**body)
#     session_todos.setdefault(session_id, []).append(todo.model_dump())
#     await event_emitter.emit_todo_created(
#         session_id=session_id,
#         todo=todo.model_dump(),
#     )
#     return todo


# @router.patch("/session/{session_id}/todo")
# async def update_todo(
#     session_id: str,
#     todo_id: str,
#     body: dict[str, Any],
#     user_id: Annotated[str, Depends(get_current_user_id)],
#     session=Depends(get_db_session),
# ):
#     """update a todo for a session."""
#     from backend.src.api.schemas.tui_control_schemas import Todo

#     todos = session_todos.get(session_id, [])
#     for i, todo in enumerate(todos):
#         if todo.get("id") == todo_id:
#             updated = Todo(**{**todo, **body})
#             todos[i] = updated.model_dump()
#             await event_emitter.emit_todo_updated(
#                 session_id=session_id,
#                 todo=updated.model_dump(),
#             )
#             return updated
#     raise HTTPException(status_code=404, detail="Todo not found")


# @router.delete("/session/{session_id}/todo/{todo_id}")
# async def delete_todo(
#     session_id: str,
#     todo_id: str,
#     user_id: Annotated[str, Depends(get_current_user_id)],
#     session=Depends(get_db_session),
# ):
#     """delete a todo for a session."""
#     todos = session_todos.get(session_id, [])
#     for i, todo in enumerate(todos):
#         if todo.get("id") == todo_id:
#             deleted = todos.pop(i)
#             await event_emitter.emit_todo_deleted(
#                 session_id=session_id,
#                 todo=deleted,
#             )
#             return deleted
#     raise HTTPException(status_code=404, detail="Todo not found")


# @router.post("/session/{session_id}/todos")
# async def create_todos(
#     session_id: str,
#     body: list[dict[str, Any]],
#     user_id: Annotated[str, Depends(get_current_user_id)],
#     session=Depends(get_db_session),
# ):
#     """create multiple todos for a session."""
#     from backend.src.api.schemas.tui_control_schemas import Todo

#     created = []
#     for todo_body in body:
#         todo = Todo(**todo_body)
#         session_todos.setdefault(session_id, []).append(todo.model_dump())
#         created.append(todo.model_dump())
#         await event_emitter.emit_todo_created(
#             session_id=session_id,
#             todo=todo.model_dump(),
#         )
#     return {"todos": created}


# DIFFS


@router.get("/session/{session_id}/diff")
async def get_diffs(session_id: str):
    return session_diffs.get(session_id, [])


# CONFIG


@router.get("/config")
async def get_config():
    return get_default_config()


@router.get("/config/providers")
async def get_config_providers():
    return get_config_providers_response()


# PROVIDERS


@router.get("/provider")
async def list_providers():
    return get_provider_list_response()


@router.get("/provider/auth")
async def provider_auth():
    """return which providers have API keys configured."""
    from backend.src.api.schemas.provider_registry import _build_providers

    result = {}
    for p in _build_providers():
        if p.env:
            result[p.id] = any(os.environ.get(var) for var in p.env)
        else:
            result[p.id] = True
    return result


@router.post("/provider/auth")
async def set_provider_auth(body: dict[str, Any]) -> bool:  # type: ignore[reportExplicitAny]
    """store an API key for a provider in .env and os.environ."""
    provider_id = body.get("providerID")  # type: ignore[reportAny]
    auth = body.get("auth", {})  # type: ignore[reportAny]
    key = auth.get("key")  # type: ignore[reportAny]
    if not provider_id or not key:
        raise HTTPException(status_code=400, detail="providerID and auth.key required")

    from backend.src.api.schemas.provider_registry import _build_providers

    provider = next((p for p in _build_providers() if p.id == provider_id), None)
    if not provider or not provider.env:
        raise HTTPException(
            status_code=404,
            detail=f"Provider {provider_id} not found or has no env vars",
        )

    env_var = provider.env[0]

    # set in current process
    os.environ[env_var] = key  # type: ignore[reportAny]

    # persist to .env file
    env_path = os.path.join(PROJECT_DIR, ".env")
    _update_env_file(env_path, env_var, key)  # type: ignore[reportAny]

    return True


def _update_env_file(env_path: str, key: str, value: str) -> None:
    """add or update a key=value pair in a .env file."""
    lines: list[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)


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


@router.post("/lsp/updated")
async def emit_lsp_updated(
    body: dict[str, Any],  # type: ignore[reportExplicitAny]
) -> dict[str, bool]:
    """
    emit lsp.updated to all SSE subscribers.
    body: arbitrary LSP state dict — the TUI consumes this to refresh diagnostics.
    """
    await event_emitter.emit_lsp_updated(body)  # type: ignore[reportUnknownMemberType]
    return {"ok": True}


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

_last_vcs_branch: str | None = None


@router.get("/vcs")
async def vcs_info():
    global _last_vcs_branch
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        branch = "main"

    if _last_vcs_branch is not None and _last_vcs_branch != branch:
        await event_emitter.emit_vcs_branch_updated({"branch": branch})
    _last_vcs_branch = branch
    return {"branch": branch, "dirty": False}


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
