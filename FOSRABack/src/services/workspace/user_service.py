from __future__ import annotations

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from FOSRABack.src.api.schemas import (
    ConvoRequest,
    NewUserRequest,
    UserRequest,
    UserRequestBase,
    UserUpdateRequest,
    UserResponse,
)

from FOSRABack.src.api.schemas.api_schemas import NewUserResponse
from FOSRABack.src.domain.exceptions import (
    UserExistenceError,
    UserRetrievalError,
    UserStorageError,
)
from FOSRABack.src.domain.schemas import User
from FOSRABack.src.domain.schemas.schemas import UserLogin, UserUpdate
from FOSRABack.src.storage.repos.user_repo import UserRepo
from FOSRABack.src.storage.utils.converters import (
    domain_to_response,
    pydantic_to_domain,
)


class UserService:
    @staticmethod
    async def get_or_create_default_user(
        user_request: UserRequest | NewUserRequest,
        session: AsyncSession,
    ) -> UserResponse:
        try:
            user: User = await UserRepo().get_or_create_user(
                user_request=user_request, session=session
            )

            return domain_to_response(user, UserResponse)

        except Exception:
            raise

    @staticmethod
    async def retrieve_user_by_id(
        user_request: UserRequest, session: AsyncSession
    ) -> UserResponse:
        try:
            user = await UserRepo().retrieve_user_by_id(
                session=session, user_id=user_request.user_id
            )

            return domain_to_response(user, UserResponse)

        except Exception as e:
            logger.error(f"Error retrieving user {user_request.user_id}: {e}")
            raise UserRetrievalError(
                user_id=user_request.user_id,
                reason=f"Failed to retrieve user: {e}",
                remediation="Verify the user ID is correct and exists.",
            ) from e

    @staticmethod
    async def get_all_users(
        session: AsyncSession,
    ) -> list[UserResponse]:
        users = await UserRepo.get_all_users(
            session=session,
        )

        list_users = [domain_to_response(user, UserResponse) for user in users]
        return list_users

    @staticmethod
    async def create_user(
        create_user: NewUserRequest,
        session: AsyncSession,
    ) -> UserResponse:
        user = await UserRepo.create_user(new_user=create_user, session=session)
        logger.info(f"Created user {user.user_id} for user {user.username}")

        return domain_to_response(user, UserResponse)

    @staticmethod
    async def update_user(
        user_update: UserUpdateRequest,
        session: AsyncSession,
    ) -> UserResponse:
        user_domain = pydantic_to_domain(user_update, UserUpdate)

        user: User = await UserRepo.update_user(
            user_update=user_domain, session=session
        )

        logger.info(f"Updated user {user.user_id} for user {user.user_id}")
        return domain_to_response(user, UserResponse)

    @staticmethod
    async def delete_user(
        user_request: UserRequest,
        session: AsyncSession,
    ) -> bool:
        result = await UserRepo.delete_user(
            user_id=user_request.user_id, session=session
        )
        logger.info(
            f"Deleted user {user_request.user_id} for user {user_request.user_id}"
        )
        return result

    @staticmethod
    async def delete_list_of_users(
        user_list: list[str],
        session: AsyncSession,
    ) -> bool:
        result = False

        try:
            if user_list:
                u_list = user_list
                for id in u_list:
                    result = await UserRepo.delete_user(user_id=id, session=session)
                    logger.info(f"Deleted user {id}!")
            return result

        except Exception:
            raise

    @staticmethod
    async def user_exists(user_request: UserRequest, session: AsyncSession) -> bool:
        exists = await UserRepo.exists(user_id=user_request.user_id, session=session)
        return exists

    @staticmethod
    async def check_user_exists(username, session: AsyncSession) -> bool:
        try:
            exists = await UserRepo().validate_user(session=session, username=username)

            return exists

        except Exception as e:
            raise

    @staticmethod
    async def check_user_login(
        username: str, password: str, session: AsyncSession
    ) -> User:
        try:
            user = await UserRepo().validate_login(
                session=session, username=username, password=password
            )

            return user

        except Exception as e:
            raise
