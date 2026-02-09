from datetime import UTC, datetime
from typing import Any, Iterator, TypeVar, Type, cast, dataclass_transform
from fastapi import HTTPException
from loguru import logger
import msgspec
from pydantic import BaseModel
from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase
from ulid import ULID

T_Pydantic = TypeVar("T_Pydantic", bound=BaseModel)
T_Msgspec = TypeVar("T_Msgspec", bound=msgspec.Struct)
T_ORM = TypeVar("T_ORM", bound=DeclarativeBase)


def pydantic_to_domain(
    pydantic_obj: T_Pydantic, domain_cls: Type[T_Msgspec]
) -> T_Msgspec:
    return msgspec.convert(
        pydantic_obj.model_dump(),
        type=domain_cls,
    )


def domain_to_orm(domain_obj: T_Msgspec, orm_class: Type[T_ORM]) -> T_ORM:
    """
    STORAGE: Domain Model (msgspec) -> SQLAlchemy ORM
    """
    data = msgspec.to_builtins(domain_obj)
    return orm_class(**data)


def orm_to_domain(orm_instance: T_ORM, domain_cls: Type[T_Msgspec]) -> T_Msgspec:
    try:
        data = _orm_to_safe_dict(orm_instance)

        return msgspec.convert(
            data,
            type=domain_cls,
            from_attributes=True,
            strict=False,
        )
    except Exception as e:
        logger.error(f"Error converting ORM to domain: {e}")
        raise


def _orm_to_safe_dict(orm_instance: DeclarativeBase) -> dict[str, Any] | None:
    if orm_instance is None:
        return None

    inspector = inspect(orm_instance)

    data = inspector.dict.copy()

    _handle_metadata_mapping(orm_instance, data)

    class_name = orm_instance.__class__.__name__

    if class_name == "ConvoORM":
        data = _handle_convo_relationships(orm_instance, data, inspector)
    elif class_name == "MessageORM":
        pass
    else:
        logger.debug(f"Skipping relationships for {class_name} to avoid lazy loading")

    return data


def _handle_metadata_mapping(
    orm_instance: DeclarativeBase, data: dict[str, Any]
) -> None:
    metadata_keys = ["convo_metadata", "message_metadata", "metadata"]

    for key in metadata_keys:
        if key in data:
            if key != "metadata":
                data["metadata"] = data.pop(key)
            break


def _handle_convo_relationships(
    convo_orm: DeclarativeBase, data: dict[str, Any], inspector: Any
) -> dict[str, Any]:
    from backend.src.storage.models import ConvoORM

    if "messages" not in inspector.unloaded and hasattr(convo_orm, "messages"):
        data["messages"] = []

        for message in convo_orm.messages:
            msg_inspector = inspect(message)
            msg_data = msg_inspector.dict.copy()

            if "message_metadata" in msg_data:
                msg_data["metadata"] = msg_data.pop("message_metadata")

            data["messages"].append(msg_data)

    if "user" not in inspector.unloaded and hasattr(convo_orm, "user"):
        user_inspector = inspect(convo_orm.user)
        user_data = user_inspector.dict.copy()

        safe_user_data = {
            "user_id": user_data.get("user_id"),
            "username": user_data.get("username"),
            "enabled": user_data.get("enabled"),
            "created_at": user_data.get("created_at"),
            "last_login": user_data.get("last_login"),
        }
        data["user"] = safe_user_data

    if "workspace" not in inspector.unloaded and hasattr(convo_orm, "workspace"):
        ws_inspector = inspect(convo_orm.workspace)
        ws_data = ws_inspector.dict.copy()

        safe_ws_data = {
            "workspace_id": ws_data.get("workspace_id"),
            "name": ws_data.get("name"),
            "description": ws_data.get("description"),
            "user_id": ws_data.get("user_id"),
            "archived_convos": ws_data.get("archived_convos"),
        }
        data["workspace"] = safe_ws_data

    return data


def domain_to_response(
    domain_obj: T_Msgspec, response_cls: Type[T_Pydantic]
) -> T_Pydantic:
    """EGRESS: Domain Model (msgspec) -> API Response (Pydantic)"""

    logger.debug(f"Converting {type(domain_obj).__name__}")
    logger.debug(f"Struct fields: {domain_obj.__struct_fields__}")

    try:
        data = msgspec.to_builtins(domain_obj)
        logger.debug(f"Converted to builtins: {type(data)}")
    except Exception as e:
        logger.error(f"Failed on field inspection:")
        for field_name in domain_obj.__struct_fields__:
            field_val = getattr(domain_obj, field_name)
            logger.error(f"  {field_name}: {type(field_val)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error converting domain object: {e}",
        )

    return response_cls.model_validate(data)


@dataclass_transform()
class DomainStruct(msgspec.Struct):
    def __repr__(self) -> str:
        fields = ", ".join(f"{f}={getattr(self, f)!r}" for f in self.__struct_fields__)
        return f"{self.__class__.__name__}({fields})"

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from self.__dict__.items()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def to_orm(self, orm_class: Type[T_ORM]) -> T_ORM:
        return domain_to_orm(self, orm_class)

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ulid_factory() -> str:
    return str(ULID())


def utc_now() -> datetime:
    return datetime.now(UTC)
