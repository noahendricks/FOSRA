"""
provider routes.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/provider", tags=["Provider"])


@router.get("/")
async def list_providers():
    """List all providers."""
    return {
        "all": [],
        "default": {},
        "connected": [],
    }


@router.get("/auth")
async def get_provider_auth():
    """Get provider auth methods."""
    return {}


@router.post("/{provider_id}/oauth/authorize")
async def oauth_authorize(provider_id: str, body: dict):
    """Initiate OAuth authorization."""
    return None


@router.post("/{provider_id}/oauth/callback")
async def oauth_callback(provider_id: str, body: dict):
    """Handle OAuth callback."""
    return True
