"""
health, global config, and dispose routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Global"])


@router.get("/health")
async def health():
    """Get health information."""
    return {"healthy": True, "version": "1.0.0"}


@router.get("/config")
async def get_global_config():
    """Get global configuration."""
    return {}


@router.patch("/config")
async def update_global_config(config: dict):
    """Update global configuration."""
    return config


@router.post("/dispose")
async def dispose():
    """Dispose all instances."""
    return True
