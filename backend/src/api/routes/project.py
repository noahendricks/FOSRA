"""
project routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/project", tags=["Project"])


@router.get("/")
async def list_projects():
    """List all projects."""
    return []


@router.get("/current")
async def get_current_project():
    """Get current project."""
    return {}


@router.post("/git/init")
async def init_git():
    """Initialize git repository."""
    return {}


@router.patch("/{project_id}")
async def update_project(project_id: str, body: dict):
    """Update project properties."""
    return {}
