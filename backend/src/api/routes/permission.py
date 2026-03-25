"""
permission routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/permission", tags=["Permission"])


@router.get("/")
async def list_permissions():
    """List pending permissions."""
    return []


@router.post("/{request_id}/reply")
async def reply_to_permission(request_id: str, body: dict):
    """Respond to permission request."""
    return True
