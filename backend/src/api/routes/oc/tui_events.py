"""
tui event publisher routes.

routes the tui calls to trigger ui actions: append-prompt, execute-command,
show-toast, select-session, etc. plus the control loop infrastructure.

these routes publish events to the global event bus; the tui subscribes
to /oc/event via sse and receives them as json events.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.src.api.events import event_bus

router = APIRouter(prefix="/oc/tui", tags=["TUI Events"])

_control_request_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
_control_response_queues: dict[str, asyncio.Queue] = {}


class AppendPromptBody(BaseModel):
    text: str


class ExecuteCommandBody(BaseModel):
    command: str


class ShowToastBody(BaseModel):
    title: str | None = None
    message: str
    variant: str = "info"
    duration: int | None = None


class SelectSessionBody(BaseModel):
    sessionID: str


class PublishBody(BaseModel):
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ControlResponseBody(BaseModel):
    request_id: str
    body: dict[str, Any]


@router.post("/append-prompt")
async def append_prompt(body: AppendPromptBody) -> bool:
    await event_bus.publish(
        {
            "type": "tui.prompt.append",
            "properties": {"text": body.text},
        }
    )
    return True


@router.post("/submit-prompt")
async def submit_prompt() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "prompt.submit"},
        }
    )
    return True


@router.post("/clear-prompt")
async def clear_prompt() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "prompt.clear"},
        }
    )
    return True


@router.post("/execute-command")
async def execute_command(body: ExecuteCommandBody) -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": body.command},
        }
    )
    return True


@router.post("/show-toast")
async def show_toast(body: ShowToastBody) -> bool:
    await event_bus.publish(
        {
            "type": "tui.toast.show",
            "properties": {
                "title": body.title,
                "message": body.message,
                "variant": body.variant,
                "duration": body.duration,
            },
        }
    )
    return True


@router.post("/select-session")
async def select_session(body: SelectSessionBody) -> bool:
    await event_bus.publish(
        {
            "type": "tui.session.select",
            "properties": {"sessionID": body.sessionID},
        }
    )
    return True


@router.post("/publish")
async def publish(body: PublishBody) -> bool:
    await event_bus.publish(
        {
            "type": body.type,
            "properties": body.properties,
        }
    )
    return True


@router.post("/open-help")
async def open_help() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "help"},
        }
    )
    return True


@router.post("/open-sessions")
async def open_sessions() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "session.list"},
        }
    )
    return True


@router.post("/open-themes")
async def open_themes() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "theme"},
        }
    )
    return True


@router.post("/open-models")
async def open_models() -> bool:
    await event_bus.publish(
        {
            "type": "tui.command.execute",
            "properties": {"command": "model"},
        }
    )
    return True


# CONTROL LOOP — long-poll / request-response pattern for tui → backend dialogs


@router.get("/control/next")
async def control_next(request: Request):
    """
    long-poll endpoint. the tui calls this to retrieve pending control requests
    (e.g. permission.dialog, question.dialog) from the backend.

    returns {path, body, request_id} when a request is pending,
    or 204 no content when timeout is reached.
    """
    try:
        req = await asyncio.wait_for(
            _control_request_queue.get(),
            timeout=30.0,
        )
        return req
    except asyncio.TimeoutError:
        return {"status": 204}


@router.post("/control/response")
async def control_response(body: ControlResponseBody) -> bool:
    """
    the tui calls this to respond to a pending control request.
    resolves the future stored for request_id.
    """
    q = _control_response_queues.get(body.request_id)
    if q:
        await q.put(body.body)
        return True
    return False
