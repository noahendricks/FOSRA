from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from backend.src.api.schemas.base import BaseModelFlex
from backend.src.settings import UserPreferences

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RequestContext(BaseModelFlex):
    user_id: str
    session_id: str | None = None

    preferences: UserPreferences = Field(
        default=UserPreferences(),
        repr=False,
    )

    @classmethod
    async def from_request(
        cls,
        user_id: str,
        session_id: str | None,
        session: AsyncSession,
    ) -> RequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            preferences=UserPreferences(),
        )

    @classmethod
    def create_simple(
        cls,
        user_id: str,
        session_id: str | None = None,
        preferences: UserPreferences | None = None,
    ) -> RequestContext:
        return cls(
            user_id=user_id,
            session_id=session_id,
            preferences=preferences or UserPreferences(),
        )

    @classmethod
    def create_anonymous(cls) -> RequestContext:
        return cls(
            user_id="system",
            preferences=UserPreferences(),
        )


OptionalContext = RequestContext | None
