"""
session routes.
"""

from __future__ import annotations

from typing import Annotated, Any
from fastapi import APIRouter, Depends, Query

from backend.src.api.dependencies import get_current_user_id, get_db_session

router = APIRouter(prefix="/session", tags=["Session"])


@router.get("/")
async def list_sessions(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
    directory: str | None = Query(None),
    roots: bool = Query(False),
    start: int | None = Query(None),
    search: str | None = Query(None),
    limit: int | None = Query(None),
):
    """List sessions with optional filtering."""
    return []


@router.get("/status")
async def session_status():
    """Get session status for all sessions."""
    return {}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    """Get a specific session."""
    return {}


@router.get("/{session_id}/children")
async def get_session_children(session_id: str):
    """Get child sessions."""
    return []


@router.get("/{session_id}/todo")
async def get_session_todos(session_id: str):
    """Get session todos."""
    return []


@router.post("/")
async def create_session(
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
    body: dict | None = None,
):
    """Create a new session."""
    return {}


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    """Delete a session."""
    return True


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    body: dict,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
):
    """Update session properties."""
    return {}


@router.post("/{session_id}/init")
async def init_session(
    session_id: str,
    body: dict | None = None,
):
    """Initialize session (create AGENTS.md)."""
    return True


@router.post("/{session_id}/fork")
async def fork_session(
    session_id: str,
    body: dict | None = None,
):
    """Fork a session."""
    return {}


@router.post("/{session_id}/abort")
async def abort_session(session_id: str):
    """Abort an active session."""
    return True


@router.post("/{session_id}/share")
async def share_session(session_id: str):
    """Share a session."""
    return {}


@router.get("/{session_id}/diff")
async def get_session_diff(
    session_id: str,
    message_id: str = Query(...),
):
    """Get message diff."""
    return []


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str):
    """Unshare a session."""
    return {}


@router.post("/{session_id}/summarize")
async def summarize_session(
    session_id: str,
    body: dict,
):
    """Summarize a session."""
    return True


@router.get("/{session_id}/message")
async def get_session_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session=Depends(get_db_session),
    limit: int | None = Query(None),
    before: str | None = Query(None),
):
    """Get session messages (paginated)."""
    return []


@router.get("/{session_id}/message/{message_id}")
async def get_message(
    session_id: str,
    message_id: str,
):
    """Get a specific message."""
    return {}


@router.delete("/{session_id}/message/{message_id}")
async def delete_message(
    session_id: str,
    message_id: str,
):
    """Delete a message."""
    return True


@router.delete("/{session_id}/message/{message_id}/part/{part_id}")
async def delete_message_part(
    session_id: str,
    message_id: str,
    part_id: str,
):
    """Delete a message part."""
    return True


@router.patch("/{session_id}/message/{message_id}/part/{part_id}")
async def update_message_part(
    session_id: str,
    message_id: str,
    part_id: str,
    body: dict,
):
    """Update a message part."""
    return {}


@router.post("/{session_id}/message")
async def send_message(
    session_id: str,
    body: dict,
):
    """Send a message (streaming response)."""
    return {}


@router.post("/{session_id}/prompt_async")
async def send_async_message(
    session_id: str,
    body: dict,
):
    """Send an async message."""
    return None


@router.post("/{session_id}/command")
async def send_command(
    session_id: str,
    body: dict,
):
    """Send a command."""
    return {}


@router.post("/{session_id}/shell")
async def run_shell(
    session_id: str,
    body: dict,
):
    """Run a shell command."""
    return {}


@router.post("/{session_id}/revert")
async def revert_message(
    session_id: str,
    body: dict,
):
    """Revert a message."""
    return {}


@router.post("/{session_id}/unrevert")
async def unrevert_messages(session_id: str):
    """Restore reverted messages."""
    return {}


@router.post("/{session_id}/permissions/{permission_id}")
async def respond_to_permission(
    session_id: str,
    permission_id: str,
    body: dict,
):
    """Respond to permission request (deprecated)."""
    return True
