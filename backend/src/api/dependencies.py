from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import TaskiqDepends

from backend.src.api.lifecycle import Infrastructure


import os

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

DEV_USER_ID = "dev-user-001"
DEV_USER_NAME = "Development User"


async def get_infra(
    request: Annotated[Request, TaskiqDepends()],
) -> Infrastructure:
    if request.app.state and hasattr(request.app.state, "infra"):
        return request.app.state.infra
    else:
        raise RuntimeError("Infrastructure not found in application state")


async def get_db_session(
    infra: Annotated[Infrastructure, Depends(get_infra)],
) -> AsyncGenerator[AsyncSession, None]:
    if not infra or not infra.session_factory:
        raise RuntimeError("Infrastructure or session factory not initialized")

    async with infra.session_factory() as session:
        yield session


async def get_session_factory(
    infra: Annotated[Infrastructure, Depends(get_infra)],
) -> async_sessionmaker[AsyncSession]:
    if not infra or not infra.session_factory:
        raise RuntimeError("Infrastructure or session factory not initialized")

    return infra.session_factory


async def get_current_user_id(
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> str:
    if not AUTH_ENABLED:
        logger.warning("Auth disabled — using dev user for all requests")
        return DEV_USER_ID

    if x_user_id:
        logger.warning("Using X-User-ID header — not for production use")
        return x_user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
