"""
permission routes: list pending permission requests, reply to a request.

the tui displays permission prompts and the user replies via these endpoints.
the reply resolves the asyncio.Future that the agent is waiting on.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.dependencies import get_current_user_id
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.api.routes.oc.state import (
    pending_permissions,
    permission_requests,
)
from backend.src.api.schemas.tui_control_schemas import (
    PermissionAction,
    PermissionRequest,
)
from loguru import logger

router = APIRouter(prefix="/oc/permission", tags=["Permission"])
event_emitter = get_event_emitter()


# @router.get("")
# async def list_permissions(
#     user_id: Annotated[str, Depends(get_current_user_id)],
# ) -> list[PermissionRequest]:
#     """
#     return all pending permission requests across all sessions for the current user.
#     the tui filters by sessionID on the client side.
#     """
#     all_requests: list[PermissionRequest] = []
#     for session_id, requests in cast(
#         dict[str, list[PermissionRequest]], permission_requests
#     ).items():
#         for req in requests:
#             all_requests.append(req)
#     return all_requests


@router.post("/{request_id}/reply")
async def reply_permission(
    request_id: str,
    body: PermissionAction,
) -> bool:
    """
    reply to a permission request.
    body: { "reply": "once" | "always" | "reject" }
    resolves the pending asyncio.Future so the agent can continue.
    publishes permission.replied event.
    """
    if body.reply not in ("once", "always", "reject"):
        raise HTTPException(
            status_code=400, detail="reply must be 'once', 'always', or 'reject'"
        )

    future = cast("asyncio.Future[str] | None", pending_permissions.get(request_id))
    if future is None:
        raise HTTPException(status_code=404, detail="Permission request not found")

    for session_id, requests in cast(
        dict[str, list[PermissionRequest]], permission_requests
    ).items():
        permission_requests[session_id] = [r for r in requests if r.id != request_id]

    if not future.done():
        future.set_result(body.reply)

    pending_permissions.pop(request_id, None)

    if body.message:
        logger.info("Permission reject with feedback: {}", body.message)

    await event_emitter.emit_permission_replied(
        "",
        request_id,
        body.reply,
    )
    return True
