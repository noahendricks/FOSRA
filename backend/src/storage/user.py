from __future__ import annotations

from datetime import datetime, UTC

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.src.api.schemas import UserRequest
from backend.src.api.schemas.api_schemas import NewUserRequest
from backend.src.domain.exceptions import (
    UserExistenceError,
    UserRetrievalError,
    UserStorageError,
)

from backend.src.domain.schemas.schemas import UserLogin, UserUpdate
from backend.src.storage.models import UserORM, ulid_factory
from backend.src.storage.utils.converters import orm_to_domain
from backend.src.domain.schemas import (
    User,
)


# =========================================================================
# CRUD Operations
# =========================================================================


class UserRepo:
    @staticmethod
    async def retrieve_user_by_id(session: AsyncSession, user_id: str) -> User:
        try:
            result = await session.execute(
                select(UserORM).where(UserORM.user_id == user_id)
            )

            user_orm = result.scalar_one_or_none()

            if not user_orm:
                logger.warning(f"User {user_id} not found.")

                raise UserRetrievalError(
                    f"User {user_id} not found",
                    remediation="Check User ID",
                    reason="User does not exist",
                )

            logger.success(f"User {user_id} retrieved.")

            return orm_to_domain(user_orm, User)

        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            raise UserRetrievalError(
                "Failed to retrieve user",
                remediation="Check User ID and Database Connection",
                reason=str(e),
            ) from e

    @staticmethod
    async def validate_user(session: AsyncSession, username: str) -> bool:
        logger.success("in validate user")

        result = await session.execute(
            select(UserORM).where(UserORM.username == username.lower())
        )

        try:
            user_orm = result.scalar_one_or_none()

            if not user_orm:
                logger.success("was None")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"exception: {e}")
            return False

    @staticmethod
    async def validate_login(
        session: AsyncSession, username: str, password: str
    ) -> User:
        result = await session.execute(
            select(UserORM).where(UserORM.username == username.lower())
        )

        try:
            user_orm = result.scalar_one_or_none()

            if not user_orm:
                raise ValueError("User Doesn't Exist")

            if user_orm.password == password:
                return orm_to_domain(user_orm, User)
            else:
                raise ValueError("Password is incorrect")

        except Exception as e:
            raise RuntimeError(e)

    @staticmethod
    async def create_user(
        session,
        new_user: NewUserRequest,
    ) -> User:
        _name = new_user.username.lower()

        try:
            user_orm = UserORM(
                username=_name,
                password=new_user.password,
                enabled=new_user.enabled,
            )
            session.add(user_orm)

            await session.commit()
            await session.refresh(user_orm)

            logger.success(f"Created user {user_orm.user_id}")
            return orm_to_domain(user_orm, User)

        except Exception as e:
            logger.error(f"Error creating user {new_user.username}: {e}")
            raise UserStorageError(
                operation="create user",
                user_id="N/A",
                remediation="Ensure User Data is Formatted Correctly and is Present",
                reason=str(e),
            ) from e

    @staticmethod
    async def get_or_create_user(
        user_request: UserRequest | NewUserRequest,
        session: AsyncSession,
    ) -> User:
        workspace = None

        try:
            if isinstance(user_request, UserRequest) and user_request.user_id:
                user = await UserRepo().retrieve_user_by_id(
                    session=session, user_id=user_request.user_id
                )
                if not user:
                    raise UserExistenceError(
                        user_id=user_request.user_id,
                        remediation="Create the user before retrieval.",
                    )

                return user

            elif isinstance(user_request, NewUserRequest):
                create_data = NewUserRequest(
                    **user_request.model_dump(exclude={"user_id"}),
                    username="Default User",
                )

                workspace = await UserRepo.create_user(
                    new_user=create_data, session=session
                )
                return workspace
            else:
                raise UserStorageError(
                    user_id=user_request.user_id,
                    operation="GetOrCreateDefaultUser",
                    reason="Invalid workspace request provided.",
                    remediation="Provide a valid workspace ID or creation data.",
                )

        except (
            UserExistenceError,
            UserRetrievalError,
            UserStorageError,
            Exception,
        ) as e:
            logger.error(f"Error retrieving workspace: {e}")
            raise

    @staticmethod
    async def update_user(
        session,
        user_update: UserUpdate,
    ) -> User:
        try:
            result = await session.execute(
                select(UserORM).where(UserORM.user_id == user_update.user_id)
            )

            user_orm = result.scalar_one_or_none()

            if not user_orm:
                raise ValueError(f"User {user_update.user_id} not found")

            update_dict = user_update.to_dict()

            for key, value in update_dict.items():
                if hasattr(user_orm, key):
                    setattr(user_orm, key, value)

            await session.commit()
            await session.refresh(user_orm)

            logger.success(f"Updated user {user_update.user_id}")
            return orm_to_domain(user_orm, User)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error updating user {user_update.user_id}: {e}")
            raise UserStorageError(
                operation="update user",
                user_id=user_update.user_id,
                reason=f"{str(e)}",
                remediation="Ensure Requested User Exists and the New Data (Update Data) is formatted correctly and present",
            ) from e

    @staticmethod
    async def delete_user(session: AsyncSession, user_id: str) -> bool:
        try:
            result = await session.execute(
                select(UserORM).where(UserORM.user_id == user_id)
            )
            user_orm = result.scalar_one_or_none()

            if not user_orm:
                raise ValueError(f"User {user_id} not found")

            await session.delete(user_orm)
            await session.commit()

            logger.info(f"Deleted user {user_id}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            raise UserStorageError(
                user_id=user_id,
                operation="delete user",
                reason=f"Failed to delete user: {e}",
                remediation="Ensure Requested User Exists",
            ) from e

    @staticmethod
    async def exists(session: AsyncSession, user_id: str) -> bool:
        try:
            result = await session.execute(
                select(UserORM.user_id).where(UserORM.user_id == user_id)
            )
            return result.scalar_one_or_none() is not None

        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
            return False

    @staticmethod
    async def update_last_login(session, user_id: str) -> None:
        try:
            result = await session.execute(
                select(UserORM).where(UserORM.user_id == user_id)
            )
            user_orm = result.scalar_one_or_none()

            if user_orm:
                user_orm.last_login = datetime.now(UTC)
                await session.commit()

        except Exception as e:
            logger.warning(f"Failed to update last_login for {user_id}: {e}")

    @staticmethod
    async def get_all_users(
        session: AsyncSession,
    ) -> list[User]:
        try:
            result = await session.execute(select(UserORM))

            user_orms = result.scalars().all()

            users = [orm_to_domain(user_orm, User) for user_orm in user_orms]

            logger.info(f"Retrieved {len(users)} users.")
            return users

        except Exception as e:
            logger.error(f"Error retrieving all users: {e}")
            raise UserRetrievalError(
                "Failed to retrieve users",
                remediation="Check Database Connection",
                reason=str(e),
            ) from e
