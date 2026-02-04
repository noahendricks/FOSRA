from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from FOSRABack.src.api.schemas.config_api_schemas import (
    EmbedderConfigRequest,
    LLMConfigRequest,
    ParserConfigRequest,
    RerankerConfigRequest,
    StorageConfigRequest,
    VectorStoreConfigRequest,
    WorkspacePreferencesAPI,
)
from FOSRABack.src.domain.enums import StorageBackendType
from FOSRABack.src.domain.exceptions import StorageError


from FOSRABack.src.storage.models import (
    ConfigAssignmentORM,
    ConfigRole,
    ToolCategory,
    UserToolConfigORM,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RequestContext(BaseModel):
    user_id: str
    workspace_id: str
    convo_id: str | None = None

    preferences: WorkspacePreferencesAPI = Field(
        default=WorkspacePreferencesAPI(),
        repr=False,
    )

    @classmethod
    async def from_request(
        cls,
        user_id: str,
        workspace_id: str,
        convo_id: str | None,
        session: AsyncSession,
    ) -> RequestContext:
        preferences = await cls._load_preferences(
            user_id=user_id,
            workspace_id=workspace_id,
            convo_id=convo_id,
            session=session,
        )

        return cls(
            user_id=user_id,
            workspace_id=workspace_id,
            convo_id=convo_id,
            preferences=preferences,
        )

    @classmethod
    def create_simple(
        cls,
        user_id: str,
        workspace_id: str,
        convo_id: str | None = None,
        preferences: WorkspacePreferencesAPI | None = None,
    ) -> RequestContext:
        return cls(
            user_id=user_id,
            workspace_id=workspace_id,
            convo_id=convo_id,
            preferences=preferences or WorkspacePreferencesAPI(),
        )

    @classmethod
    def create_anonymous(cls, workspace_id: str = "") -> RequestContext:
        return cls(
            user_id="system",
            workspace_id=workspace_id,
            preferences=WorkspacePreferencesAPI(),
        )

    @classmethod
    async def _load_preferences(
        cls,
        user_id: str,
        workspace_id: str,
        convo_id: str | None,
        session: AsyncSession,
    ) -> WorkspacePreferencesAPI:
        typed_prefs = WorkspacePreferencesAPI()

        logger.error(
            f"Loading preferences for user={user_id}, workspace={workspace_id}"
        )
        try:
            from sqlalchemy import select, or_
            from sqlalchemy.orm import selectinload

            query = (
                select(ConfigAssignmentORM)
                .options(selectinload(ConfigAssignmentORM.tool_config))
                .where(
                    or_(
                        ConfigAssignmentORM.workspace_id == workspace_id,
                        ConfigAssignmentORM.convo_id == convo_id,
                    )
                )
            )

            result = await session.execute(query)
            assignments = result.scalars().all()

            workspace_map = {
                a.role: a.tool_config
                for a in assignments
                if a.workspace_id == workspace_id
            }
            convo_map = {
                a.role: a.tool_config for a in assignments if a.convo_id == convo_id
            }

            effective_configs = {**workspace_map, **convo_map}

            for role, config in effective_configs.items():
                if not config:
                    continue

                if config.category == ToolCategory.LLM:
                    typed_prefs.llms[role] = cls._tool_to_llm_schema(config)

                elif config.category == ToolCategory.VECTOR_STORE:
                    typed_prefs.vector_store = cls._tool_to_vector_schema(config)

                elif config.category == ToolCategory.EMBEDDER:
                    typed_prefs.embedder = cls._tool_to_embedder_schema(config)

                elif config.category == ToolCategory.PARSER:
                    typed_prefs.parser = cls._tool_to_parser_schema(config)

                elif config.category == ToolCategory.RERANKER:
                    typed_prefs.reranker = cls._tool_to_reranker_schema(config)
                elif config.category == ToolCategory.STORAGE:
                    typed_prefs.storage = cls._tool_to_storage_schema(config)

            return typed_prefs

        except Exception as e:
            logger.warning(f"Failed to load preferences for user {user_id}: {e}")
            raise StorageError(
                operation="load_preferences",
                reason=f"{str(e)}",
                resource_id=f"user:{user_id}-workspace:{workspace_id}",
                remediation="Ensure database is reachable and preferences exist.",
                storage_type="SQLAlchemy",
            ) from e

    @staticmethod
    def _tool_to_vector_schema(config: "UserToolConfigORM") -> VectorStoreConfigRequest:
        d = config.details
        return VectorStoreConfigRequest(
            config_id=config.id,
            config_name=config.name,
            store_type=config.provider,
            host=d.get("host", "localhost"),
            port=d.get("port", 6333),
            api_key=d.get("api_key"),
            url=d.get("url"),
            collection_name=d.get("collection_name", "default"),
            top_k=d.get("top_k", 10),
            min_score=d.get("min_score", 0.0),
            include_metadata=d.get("include_metadata", True),
        )

    @staticmethod
    def _tool_to_embedder_schema(config: "UserToolConfigORM") -> EmbedderConfigRequest:
        d = config.details
        return EmbedderConfigRequest(
            config_id=config.id,
            config_name=config.name,
            embedder_type=config.provider,
            model=config.model,
            api_key=d.get("api_key"),
            api_base=d.get("api_base"),
            batch_size=d.get("batch_size", 32),
            normalize=d.get("normalize", True),
            mode=d.get("mode", "dense"),
        )

    @staticmethod
    def _tool_to_reranker_schema(config: "UserToolConfigORM") -> RerankerConfigRequest:
        d = config.details
        return RerankerConfigRequest(
            config_id=config.id,
            config_name=config.name,
            reranker_type=config.provider,
            model=config.model,
            api_key=d.get("api_key"),
            top_k=d.get("top_k", 5),
            enabled=d.get("enabled", True),
            params=d.get("params", {}),
        )

    @staticmethod
    def _tool_to_llm_schema(config: "UserToolConfigORM") -> LLMConfigRequest:
        return LLMConfigRequest(
            config_id=config.id,
            config_name=config.name,
            provider=config.provider,
            model=config.model,
            api_key=config.details.get("api_key"),
            api_base=config.details.get("api_base"),
            litellm_params=config.details.get("params", {}),
            language=config.details.get("language", "English"),
        )

    @staticmethod
    def _tool_to_parser_schema(config: "UserToolConfigORM") -> ParserConfigRequest:
        d = config.details
        return ParserConfigRequest(
            config_id=config.id,
            config_name=config.name,
            parser_type=config.provider,
            api_key=d.get("api_key"),
            extract_tables=d.get("extract_tables", True),
        )

    @staticmethod
    def _tool_to_storage_schema(config: "UserToolConfigORM") -> StorageConfigRequest:
        """Convert UserToolConfigORM to StorageConfigRequest."""
        d = config.details
        return StorageConfigRequest(
            config_id=config.id,
            config_name=config.name,
            backend_type=d.get("backend_type", StorageBackendType.FILESYSTEM),
            timeout_seconds=d.get("timeout_seconds", 30),
            max_retries=d.get("max_retries", 3),
            chunk_size=d.get("chunk_size", 8192),
            backend_options=d.get("backend_options", {}),
            base_path=d.get("base_path"),
            api_key=d.get("api_key"),
        )

    def get_preference(self, key: str, default: Any = None) -> Any:
        return getattr(self.preferences, key, default)

    def with_new_session_id(self, convo_id: str) -> RequestContext:
        return RequestContext(
            user_id=self.user_id,
            workspace_id=self.workspace_id,
            convo_id=convo_id,
            preferences=self.preferences,
        )

    def with_new_workspace_id(self, workspace_id: str) -> RequestContext:
        return RequestContext(
            user_id=self.user_id,
            workspace_id=workspace_id,
            preferences=self.preferences,
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

    if not preferences.llm_default:
        system_tool = await get_system_default_config(session, ConfigRole.PRIMARY_LLM)
        preferences.llm_default = cls._tool_to_llm_schema(system_tool)

    return cls(
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
        preferences=preferences,
    )


async def get_system_default_config(
    session: AsyncSession, role: ConfigRole
) -> UserToolConfigORM:
    category_map = {
        ConfigRole.PRIMARY_LLM: ToolCategory.LLM,
        ConfigRole.FAST_LLM: ToolCategory.LLM,
        ConfigRole.STRATEGIC_LLM: ToolCategory.LLM,
        ConfigRole.DEFAULT_PARSER: ToolCategory.PARSER,
        ConfigRole.DEFAULT_VECTOR_STORE: ToolCategory.VECTOR_STORE,
        ConfigRole.DEFAULT_EMBEDDER: ToolCategory.EMBEDDER,
        ConfigRole.DEFAULT_RERANKER: ToolCategory.RERANKER,
    }

    target_category = category_map.get(role)

    query = (
        select(UserToolConfigORM)
        .where(
            UserToolConfigORM.user_id == "SYSTEM",
            UserToolConfigORM.category == target_category,
            UserToolConfigORM.is_system_default,
        )
        .limit(1)
    )

    result = await session.execute(query)
    system_tool = result.scalar_one_or_none()

    if not system_tool:
        raise RuntimeError(f"Critical Error: No system default configured for {role}")

    return system_tool


async def initialize_user_workspace(
    session: AsyncSession, user_id: str, workspace_id: str
):
    roles_to_init = [
        ConfigRole.PRIMARY_LLM,
        ConfigRole.FAST_LLM,
        ConfigRole.STRATEGIC_LLM,
        ConfigRole.DEFAULT_PARSER,
        ConfigRole.DEFAULT_VECTOR_STORE,
        ConfigRole.DEFAULT_EMBEDDER,
        ConfigRole.DEFAULT_RERANKER,
    ]

    assignments = []

    for role in roles_to_init:
        system_tool = await get_system_default_config(session, role)

        assignments.append(
            ConfigAssignmentORM(
                role=role,
                tool_config_id=system_tool.id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
        )

    session.add_all(assignments)

    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to initialize workspace {workspace_id}: {e}")
        raise
