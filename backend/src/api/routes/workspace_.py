"""
workspace routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/workspace", tags=["Workspace"])


@router.post("/")
async def create_workspace(body: dict):
    """Create a new workspace."""
    return {}


@router.get("/")
async def list_workspaces():
    """List all workspaces."""
    return []


@router.delete("/{workspace_id}")
async def remove_workspace(workspace_id: str):
    """Remove a workspace."""
    return True
