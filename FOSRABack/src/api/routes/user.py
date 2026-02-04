from typing import Annotated
from fastapi import APIRouter, Body, Depends, Query

from FOSRABack.src.api.dependencies import get_db_session
from FOSRABack.src.api.schemas import (
    NewUserRequest,
    UserRequest,
    UserRequestBase,
    UserResponse,
    UserUpdateRequest,
)
from FOSRABack.src.api.schemas.api_schemas import NewUserResponse, UserLogin
from FOSRABack.src.domain.schemas import User
from FOSRABack.src.services.workspace.user_service import UserService
from FOSRABack.src.storage.utils.converters import domain_to_response


router = APIRouter(prefix="/users", tags=["Users"])


# ============================================================================
# GET
# ============================================================================
@router.get("/user/profile/{user_id}")
async def get_user_profile(
    request: Annotated[UserRequest, Query()], session=Depends(get_db_session)
) -> UserResponse:
    try:
        user: UserResponse = await UserService().retrieve_user_by_id(
            user_request=request, session=session
        )
        return user
    except Exception as e:
        raise e


# ============================================================================
# POST
# ============================================================================
@router.post("/create_user")
async def new_user_profile(
    request: Annotated[NewUserRequest, Body()], session=Depends(get_db_session)
) -> UserResponse:
    try:
        new_user: UserResponse = await UserService().create_user(
            create_user=request, session=session
        )

        return new_user
    except Exception as e:
        raise e


# ============================================================================
# PUT
# ============================================================================
# PUT Update Existing User Profile / User
@router.put("/user/profile/{user_id}")
async def update_user_profile(
    request: Annotated[UserUpdateRequest, Query()], session=Depends(get_db_session)
) -> UserResponse:
    try:
        user: UserResponse = await UserService().update_user(
            user_update=request, session=session
        )

        return user
    except Exception as e:
        raise e


# ============================================================================
# DELETE
# ============================================================================
@router.delete("/user/profile/{user_id}")
async def delete_user_profile(
    request: Annotated[UserRequest, Query()], session=Depends(get_db_session)
) -> bool:
    try:
        # TODO: Ensure Cascades on Delete
        user: bool = await UserService().delete_user(
            user_request=request, session=session
        )

        return user
    except Exception as e:
        raise e


@router.get("/login/check_user")
async def check_user_exist(username: str, session=Depends(get_db_session)) -> bool:
    try:
        exists: bool = await UserService().check_user_exists(
            username=username, session=session
        )
        return exists
    except Exception as e:
        raise e


@router.get("/login/user_login")
async def validate_user_login(
    request: Annotated[UserLogin, Query()], session=Depends(get_db_session)
) -> UserResponse:
    try:
        username = request.username
        password = request.password
        user: User = await UserService().check_user_login(
            username=username, password=password, session=session
        )
        return domain_to_response(user, UserResponse)
    except Exception as e:
        raise e
