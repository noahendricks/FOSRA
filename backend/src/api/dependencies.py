from __future__ import annotations

from typing import Annotated

from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.src.api.lifecycle import Infrastructure


DEV_USER_ID = "dev-user-001"
DEV_USER_NAME = "Development User"


async def get_infra(request: Request) -> Infrastructure:
    if request.app.state and hasattr(request.app.state, "infra"):  # type: ignore[reportAny]
        return request.app.state.infra  # type: ignore[reportAny]
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


async def get_current_user_id() -> str:
    from backend.src.settings import settings

    if settings.is_development():
        return DEV_USER_ID
    raise RuntimeError(
        "get_current_user_id requires authentication in non-development environments"
    )
