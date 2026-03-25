"""
question routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/question", tags=["Question"])


@router.get("/")
async def list_questions():
    """List pending questions."""
    return []


@router.post("/{request_id}/reply")
async def reply_to_question(request_id: str, body: dict):
    """Reply to question request."""
    return True


@router.post("/{request_id}/reject")
async def reject_question(request_id: str):
    """Reject question request."""
    return True
