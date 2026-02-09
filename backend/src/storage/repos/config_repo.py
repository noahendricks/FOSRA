from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload

from backend.src.api.schemas.config_api_schemas import (
    ToolConfigCreate,
    ToolConfigUpdate,
    ToolConfigResponse,
)
from backend.src.domain.exceptions import (
    ConfigExistenceError,
    ConfigRetrievalError,
    ConfigStorageError,
)
from backend.src.storage.models import (
    UserToolConfigORM,
    ConfigAssignmentORM,
    ToolCategory,
    ConfigRole,
    ROLE_TO_CATEGORY_MAP,
)
from backend.src.storage.utils.converters import orm_to_domain

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# =========================================================================
# CRUD Operations
# =========================================================================


class ConfigRepository:
    # =========================================================================
    # Tool Config CRUD
    # =========================================================================

    @staticmethod
    async def create_tool_config(
        session: AsyncSession, user_id: str, config_data: ToolConfigCreate
    ) -> ToolConfigResponse:
        try:
            # If creating system default, ensure no existing one exists
            if config_data.is_system_default:
                existing = await ConfigRepository.get_system_default(
                    session, config_data.category
                )
                if existing:
                    raise ValueError(
                        f"System default for {config_data.category} already exists"
                    )

            config_orm = UserToolConfigORM(
                user_id=user_id,
                name=config_data.name,
                description=config_data.description,
                category=config_data.category,
                provider=config_data.provider,
                model=config_data.model,
                details=config_data.details,
                is_system_default=config_data.is_system_default,
            )

            session.add(config_orm)
            await session.commit()
            await session.refresh(config_orm)

            logger.info(f"Created tool config: {config_orm.id} ({config_orm.name})")
            return orm_to_domain(config_orm, ToolConfigResponse)

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating tool config: {e}")
            raise ConfigStorageError(
                user_id=user_id,
                config_id="",
                operation="create_tool_config",
                reason=str(e),
                remediation="Check config data and database connection",
            ) from e

    @staticmethod
    async def get_tool_config(
        session: AsyncSession, config_id: str
    ) -> ToolConfigResponse:
        try:
            result = await session.execute(
                select(UserToolConfigORM).where(UserToolConfigORM.id == config_id)
            )
            config_orm = result.scalar_one_or_none()

            if not config_orm:
                raise ConfigExistenceError(
                    user_id="",
                    config_id=config_id,
                    remediation="Verify config ID exists",
                )

            return orm_to_domain(config_orm, ToolConfigResponse)

        except ConfigExistenceError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving config {config_id}: {e}")
            raise ConfigRetrievalError(
                "Failed to retrieve config",
                user_id="",
                remediation="Check config ID and database connection",
                reason=str(e),
            ) from e

    @staticmethod
    async def list_user_configs(
        session: AsyncSession, user_id: str, category: ToolCategory | None = None
    ) -> list[ToolConfigResponse]:
        try:
            query = select(UserToolConfigORM).where(
                UserToolConfigORM.user_id == user_id
            )

            if category:
                query = query.where(UserToolConfigORM.category == category)

            result = await session.execute(query)
            configs = result.scalars().all()

            return [orm_to_domain(c, ToolConfigResponse) for c in configs]

        except Exception as e:
            logger.error(f"Error listing configs for user {user_id}: {e}")
            raise ConfigRetrievalError(
                "Failed to list configs", user_id=user_id, reason=str(e)
            ) from e

    @staticmethod
    async def update_tool_config(
        session: AsyncSession, config_id: str, update_data: ToolConfigUpdate
    ) -> ToolConfigResponse:
        try:
            result = await session.execute(
                select(UserToolConfigORM).where(UserToolConfigORM.id == config_id)
            )
            config_orm = result.scalar_one_or_none()

            if not config_orm:
                raise ConfigExistenceError(
                    user_id="", config_id=config_id, remediation="Verify config exists"
                )

            # Update fields
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(config_orm, key, value)

            await session.commit()
            await session.refresh(config_orm)

            logger.info(f"Updated config: {config_id}")
            return orm_to_domain(config_orm, ToolConfigResponse)

        except ConfigExistenceError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating config {config_id}: {e}")
            raise ConfigStorageError(
                operation="update_tool_config", config_id=config_id, reason=str(e)
            ) from e

    @staticmethod
    async def delete_tool_config(session: AsyncSession, config_id: str) -> None:
        try:
            result = await session.execute(
                select(UserToolConfigORM).where(UserToolConfigORM.id == config_id)
            )
            config_orm = result.scalar_one_or_none()

            if not config_orm:
                raise ConfigExistenceError(
                    user_id="", config_id=config_id, remediation="Verify config exists"
                )

            await session.delete(config_orm)
            await session.commit()

            logger.info(f"Deleted config: {config_id}")

        except ConfigExistenceError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error deleting config {config_id}: {e}")
            raise ConfigStorageError(
                operation="delete_tool_config", config_id=config_id, reason=str(e)
            ) from e

    # =========================================================================
    # System Defaults
    # =========================================================================

    @staticmethod
    async def get_system_default(
        session: AsyncSession, category: ToolCategory
    ) -> UserToolConfigORM | None:
        try:
            result = await session.execute(
                select(UserToolConfigORM).where(
                    UserToolConfigORM.is_system_default == True,
                    UserToolConfigORM.category == category,
                )
            )
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Error fetching system default for {category}: {e}")
            return None

    @staticmethod
    async def get_system_default_for_role(
        session: AsyncSession, role: ConfigRole
    ) -> UserToolConfigORM:
        category = ROLE_TO_CATEGORY_MAP.get(role)
        if not category:
            raise ValueError(f"Unknown role: {role}")

        config = await ConfigRepository.get_system_default(session, category)
        if not config:
            raise RuntimeError(
                f"CRITICAL: No system default configured for role {role} "
                f"(category: {category})"
            )

        return config
