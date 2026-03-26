"""
permission routes: list pending permission requests, reply to a request.

the tui displays permission prompts and the user replies via these endpoints.
the reply resolves the asyncio.Future that the agent is waiting on.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.dependencies import get_current_user_id
from backend.src.api.events import event_bus
from backend.src.api.routes.oc.state import (
    pending_permissions,
    permission_requests,
)
from backend.src.api.schemas.tui_schemas import PermissionRequest
from loguru import logger

router = APIRouter(prefix="/oc/permission", tags=["Permission"])


@router.get("")
async def list_permissions(
    user_id: Annotated[str, Depends(get_current_user_id)],
):
    """
    return all pending permission requests across all sessions for the current user.
    the tui filters by sessionID on the client side.
    """
    all_requests: list[dict[str, Any]] = []
    for session_id, requests in permission_requests.items():
        for req in requests:
            all_requests.append(req)
    return all_requests


@router.post("/{request_id}/reply")
async def reply_permission(
    request_id: str,
    body: dict[str, Any],
):
    """
    reply to a permission request.
    body: { "reply": "once" | "always" | "reject" }
    resolves the pending asyncio.Future so the agent can continue.
    publishes permission.replied event.
    """
    reply = body.get("reply")
    if reply not in ("once", "always", "reject"):
        raise HTTPException(
            status_code=400, detail="reply must be 'once', 'always', or 'reject'"
        )

    future = pending_permissions.get(request_id)
    if future is None:
        raise HTTPException(status_code=404, detail="Permission request not found")

    for session_id, requests in list(permission_requests.items()):
        permission_requests[session_id] = [
            r for r in requests if r.get("id") != request_id
        ]

    if not future.done():
        future.set_result(reply)

    pending_permissions.pop(request_id, None)

    message = body.get("message", "")
    if message:
        logger.info(f"Permission reject with feedback: {message}")

    await event_bus.publish(
        {
            "type": "permission.replied",
            "properties": {
                "sessionID": body.get("sessionID", ""),
                "requestID": request_id,
                "reply": reply,
            },
        }
    )
    return True
