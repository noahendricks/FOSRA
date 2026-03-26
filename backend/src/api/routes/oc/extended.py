"""
extended stubs: experimental routes, mcp connect/disconnect, pty, log, instance dispose.

these are properly-shaped stubs that return correct types but are not yet
functionally implemented.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Extended Stubs"])


# EXPERIMENTAL


@router.get("/experimental/resource")
async def list_experimental_resources():
    """stub — experimental resource listing."""
    return {}


@router.get("/experimental/workspace")
async def list_experimental_workspaces():
    """stub — experimental workspace listing."""
    return []


@router.post("/experimental/workspace")
async def create_experimental_workspace(body: dict):
    """stub — experimental workspace creation."""
    return {}


@router.delete("/experimental/workspace/{workspace_id}")
async def remove_experimental_workspace(workspace_id: str):
    """stub — experimental workspace removal."""
    return True


@router.get("/experimental/tool/ids")
async def list_experimental_tool_ids():
    """stub — experimental tool IDs listing."""
    return []


@router.get("/experimental/tool")
async def list_experimental_tools():
    """stub — experimental tools listing."""
    return []


# MCP


@router.get("/pty")
async def get_pty_list():
    """stub — PTY list."""
    return []


@router.post("/mcp")
async def mcp_post():
    """stub — MCP general endpoint."""
    return {}


@router.post("/mcp/{name}/connect")
async def mcp_connect(name: str):
    """stub — MCP connection."""
    return {}


@router.post("/mcp/{name}/disconnect")
async def mcp_disconnect(name: str):
    """stub — MCP disconnection."""
    return True


# LOGGING


@router.get("/log")
async def get_log():
    """stub — logging endpoint."""
    return True


# INSTANCE


@router.post("/instance/dispose")
async def dispose_instance():
    """stub — instance disposal."""
    return True
