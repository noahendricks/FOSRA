"""
project routes: list projects, get current project, init git.

project state is stored in-memory (no persistent project DB yet).
the default project is derived from PROJECT_DIR.
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, HTTPException

from backend.src.api.schemas.tui_schemas import (
    PROJECT_DIR,
    Project,
    ProjectIcon,
    ProjectSummary,
    ProjectTime,
)

router = APIRouter(prefix="/oc", tags=["Project"])

_in_memory_projects: dict[str, dict] = {}


def _default_project() -> dict:
    """build the default in-memory project from PROJECT_DIR."""
    project_id = "default"
    if project_id not in _in_memory_projects:
        _in_memory_projects[project_id] = {
            "id": project_id,
            "name": os.path.basename(PROJECT_DIR) or "FOSRA",
            "worktree": PROJECT_DIR,
            "vcs": "git",
            "icon": {"color": "#10b981"},
            "commands": {},
            "time": {
                "created": int(time.time()),
                "updated": int(time.time()),
                "initialized": None,
            },
            "sandboxes": [],
        }
    return _in_memory_projects[project_id]


@router.get("/project")
async def list_projects() -> list[ProjectSummary]:
    """return all known projects."""
    p = _default_project()
    return [
        ProjectSummary(
            id=p["id"],
            name=p["name"],
            worktree=p["worktree"],
        )
    ]


@router.get("/project/current")
async def get_current_project() -> Project:
    """return the current active project."""
    p = _default_project()
    return Project(
        id=p["id"],
        worktree=p["worktree"],
        vcs=p["vcs"],
        name=p["name"],
        icon=ProjectIcon(**p["icon"]),
        commands=p["commands"],
        time=ProjectTime(**p["time"]),
        sandboxes=p["sandboxes"],
    )


@router.patch("/project/{project_id}")
async def update_project(
    project_id: str,
    body: dict,
) -> Project:
    """update in-memory project metadata (name, icon, commands)."""
    if project_id not in _in_memory_projects:
        raise HTTPException(status_code=404, detail="Project not found")

    p = _in_memory_projects[project_id]
    if "name" in body:
        p["name"] = body["name"]
    if "icon" in body:
        p["icon"].update(body["icon"])
    if "commands" in body:
        p["commands"].update(body["commands"])
    p["time"]["updated"] = int(time.time())

    return Project(
        id=p["id"],
        worktree=p["worktree"],
        vcs=p["vcs"],
        name=p["name"],
        icon=ProjectIcon(**p["icon"]),
        commands=p["commands"],
        time=ProjectTime(**p["time"]),
        sandboxes=p["sandboxes"],
    )


@router.post("/project/git/init")
async def init_git_project(project_id: str = "default") -> bool:
    """
    run git init in the project directory.
    returns True if successful or if already a git repo.
    """
    import subprocess

    git_dir = os.path.join(PROJECT_DIR, ".git")
    if os.path.exists(git_dir):
        return True

    try:
        subprocess.check_output(
            ["git", "init"],
            cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"git init failed: {e}")
