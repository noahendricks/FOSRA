from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import TaskiqDepends

from backend.src.api.lifecycle import Infrastructure
from backend.src.api.schemas import NewUserRequest, UserRequest
from backend.src.domain.exceptions import (
    InfrastructureError,
    InitializationError,
)
from backend.src.domain.schemas import User

from backend.src.storage.models import ulid_factory
from backend.src.storage.repos.user_repo import UserRepo
from backend.src.api.request_context import RequestContext


# =============================================================================
# Stub Configuration
# =============================================================================

AUTH_ENABLED = False

DEV_USER_ID = "dev-user-001"
DEV_USER_NAME = "Development User"


# =============================================================================
# Database Session Dependencies
# =============================================================================


async def get_infra(
    request: Annotated[Request, TaskiqDepends()],
) -> Infrastructure:
    if request.app.state and hasattr(request.app.state, "infra"):
        return request.app.state.infra
    else:
        raise InitializationError(
            component_name="Infrastructure",
            reason="Infrastructure not found in application state",
        )


async def get_db_session(
    infra: Annotated[Infrastructure, Depends(get_infra)],
) -> AsyncGenerator[AsyncSession, None]:
    if not infra or not infra.session_factory:
        raise InfrastructureError(
            operation="Get DB Session",
            service_name="Database Session",
            reason="Infrastructure or session factory not initialized",
            remediation="Ensure application is properly initialized",
        )

    async with infra.session_factory() as session:
        yield session


async def get_session_factory(
    infra: Annotated[Infrastructure, Depends(get_infra)],
) -> async_sessionmaker[AsyncSession]:
    if not infra or not infra.session_factory:
        raise InfrastructureError(
            operation="Get DB Session",
            service_name="Database Session",
            reason="Infrastructure or session factory not initialized",
            remediation="Ensure application is properly initialized",
        )

    return infra.session_factory


async def get_current_user_id(
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> str:
    if not AUTH_ENABLED:
        # Development mode - use header or default
        return x_user_id or DEV_USER_ID

    if x_user_id:
        # Allow X-User-ID for testing (remove in production)
        logger.warning("Using X-User-ID header - not for production use")
        return x_user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


# async def get_or_create_current_user(
#     session: Annotated[AsyncSession, Depends(get_db_session)],
#     user_request: NewUserRequest | UserRequest,
# ) -> User:
#     try:
#         logger.debug(f"Getting or creating user: {user_request.username}")
#
#         user = await UserRepo().get_or_create_user(
#             session=session, user_request=user_request
#         )
#
#         logger.debug(f"Authenticated user: {user.user_id}")
#
#         return user
#     except Exception as e:
#         logger.error(f"Failed to get/create user {user_request.username}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Could not validate user",
#         )


async def get_optional_user(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    authorization: Annotated[str | None, Header()] = None,
    x_user_id: Annotated[str | None, Header()] = None,
) -> User | None:
    if not AUTH_ENABLED:
        user_id = x_user_id or DEV_USER_ID
    elif x_user_id:
        user_id = x_user_id
    elif authorization:
        # STUB: Extract from token
        user_id = None
    else:
        return None

    if not user_id:
        return None

    try:
        user = await UserRepo().get_or_create_user(
            session=session, user_request=UserRequest(user_id=user_id)
        )

        return user
    except Exception:
        return None


# def require_user(user: User = Depends(get_or_create_current_user)) -> User:
#     return user


# =============================================================================
# Request Context Dependencies
# =============================================================================


# async def get_request_context(
#     request: Request,
#     session: AsyncSession = Depends(get_db_session),
#     authorization: Annotated[str | None, Header()] = None,
#     x_user_id: Annotated[str | None, Header()] = None,
#     x_workspace_id: Annotated[str | None, Header()] = None,
# ) -> RequestContext:
#     if not AUTH_ENABLED:
#         user_id = x_user_id or DEV_USER_ID
#     elif x_user_id:
#         logger.warning("Using X-User-ID header - not for production use")
#         user_id = x_user_id
#     elif authorization:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="JWT authentication not implemented",
#         )
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Not authenticated",
#         )
#
#     workspace_id: str = (x_workspace_id) if x_workspace_id else ulid_factory()
#
#     convo_id: str = getattr(request.app.state, "convo_id", "")
#
#     return await RequestContext.from_request(
#         session=session,
#         workspace_id=workspace_id,
#         user_id=user_id,
#         convo_id=convo_id,
#     )
#
#
# async def get_optional_context(
#     request: Request,
#     session: AsyncSession = Depends(get_db_session),
#     authorization: Annotated[str | None, Header()] = None,
#     x_user_id: Annotated[str | None, Header()] = None,
#     x_workspace_id: Annotated[str | None, Header()] = None,
# ) -> RequestContext | None:
#     try:
#         return await get_request_context(
#             request, session, authorization, x_user_id, x_workspace_id
#         )
#     except HTTPException:
#         return None


# =============================================================================
# Utility Functions (for use outside FastAPI context)
# =============================================================================


async def authenticate_user_id(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    user_id: str,
) -> User:
    """Authenticate and get user by ID (for non-HTTP contexts)."""
    user = await UserRepo.get_or_create_user(
        session=session, user_request=UserRequest(user_id=user_id)
    )

    return user
