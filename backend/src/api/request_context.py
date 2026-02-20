from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Bundle

from backend.src.api.schemas.api_schemas import _BaseModelFlex
from backend.src.api.schemas.config_api_schemas import (
    EmbedderConfigRequest,
    LLMConfigRequest,
    ParserConfigRequest,
    RerankerConfigRequest,
    VectorStoreConfigRequest,
)
from backend.src.domain.enums import StorageBackendType
from backend.src.domain.exceptions import StorageError


from backend.src.domain.schemas import User, UserPreferences
from backend.src.storage.models import (
    ConfigRole,
    ConvoORM,
    ToolCategory,
    WorkspaceORM,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RequestContext(_BaseModelFlex):
    user_id: str
    workspace_id: str
    convo_id: str | None = None

    preferences: UserPreferences = Field(
        default=UserPreferences(),
        repr=False,
    )

    #WARN: Deprecated
    #TODO: I Need to remove this or need to fully rework this
    @classmethod
    async def from_request(
        cls,
        user_id: str,
        workspace_id: str,
        convo_id: str | None,
        session: AsyncSession,
    ) -> RequestContext:
        # NOTE:
        # stub prefs - finish later
        return cls(
            user_id=user_id,
            workspace_id=workspace_id,
            convo_id=convo_id,
            preferences=UserPreferences(),
        )

    @classmethod
    def create_simple(
        cls,
        user_id: str,
        workspace_id: str,
        convo_id: str | None = None,
        preferences: UserPreferences | None = None,
    ) -> RequestContext:
        return cls(
            user_id=user_id,
            workspace_id=workspace_id,
            convo_id=convo_id,
            preferences=preferences or UserPreferences(),
        )

    @classmethod
    def create_anonymous(cls, workspace_id: str = "") -> RequestContext:
        return cls(
            user_id="system",
            workspace_id=workspace_id,
            preferences=UserPreferences(),
        )


OptionalContext = RequestContext | None


@classmethod
async def from_request(
    cls,
    user_id: str,
    workspace_id: str,
    session: AsyncSession,
    session_id: str | None = None,
) -> RequestContext:
    preferences = await cls._load_preferences(
        user_id, workspace_id, session, session_id
    )
    #NOTE: stub - incomplete
    return cls(
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
        preferences=preferences,
    )
