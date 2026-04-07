from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, TypeVar, cast, dataclass_transform

import msgspec
from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase
from ulid import ULID

T_Pydantic = TypeVar("T_Pydantic", bound=BaseModel)
T_Msgspec = TypeVar("T_Msgspec", bound=msgspec.Struct)
T_ORM = TypeVar("T_ORM", bound=DeclarativeBase)


def pydantic_to_domain(
    pydantic_obj: T_Pydantic, domain_cls: type[T_Msgspec]
) -> T_Msgspec:
    return msgspec.convert(
        pydantic_obj.model_dump(),
        type=domain_cls,
    )


def domain_to_orm(domain_obj: T_Msgspec, orm_class: type[T_ORM]) -> T_ORM:
    data = msgspec.to_builtins(domain_obj)
    if orm_class.__name__ == "SessionORM":
        data = _map_domain_to_session_orm(data)
    return orm_class(**data)


def _map_domain_to_session_orm(data: dict[str, Any]) -> dict[str, Any]:
    if "id" in data:
        data["session_id"] = data.pop("id")

    if "time" in data:
        time_data = data.pop("time")
        if isinstance(time_data, dict):
            data["created_at"] = datetime.fromtimestamp(
                time_data.get("created", 0), UTC
            )
            data["updated_at"] = datetime.fromtimestamp(
                time_data.get("updated", 0), UTC
            )

    if "parentID" in data:
        data["parent_id"] = data.pop("parentID")

    if "metadata" in data:
        data["meta"] = data.pop("metadata")

    return data


def orm_to_domain(orm_instance: T_ORM, domain_cls: type[T_Msgspec]) -> T_Msgspec:
    try:
        data = _orm_to_safe_dict(orm_instance)

        return msgspec.convert(
            data,
            type=domain_cls,
            from_attributes=True,
            strict=False,
        )
    except Exception as e:
        logger.opt(exception=True).error("Error converting ORM to domain")
        raise


def _orm_to_safe_dict(orm_instance: DeclarativeBase | None) -> dict[str, Any] | None:
    if orm_instance is None:
        return None

    inspector = inspect(orm_instance)

    data = inspector.dict.copy()

    logger.bind(_structured={"orm safe dict inspection": data}).debug(
        "[ORM SAFE INPSECTOR DATA]"
    )

    _handle_metadata_mapping(orm_instance, data)

    class_name = orm_instance.__class__.__name__

    skipped_relationships = []

    if class_name == "SessionORM":
        data = _handle_session_relationships(orm_instance, data, inspector)
        data = _map_session_orm_to_domain(data)
    elif class_name == "MessageORM":
        pass
    else:
        if len(skipped_relationships) <= 10:
            skipped_relationships.append(
                f"Skipping relationships for {class_name} to avoid lazy loading"
            )
        else:
            logger.bind(_structured={"relationships": skipped_relationships}).debug(
                "Skipped Relationships"
            )
            skipped_relationships.clear()

    return data


def _handle_metadata_mapping(
    orm_instance: DeclarativeBase, data: dict[str, Any]
) -> None:
    metadata_keys = ["session_metadata", "message_metadata", "metadata"]

    for key in metadata_keys:
        if key in data:
            if key != "metadata":
                data["metadata"] = data.pop(key)
            break


def _handle_session_relationships(
    session_orm: DeclarativeBase, data: dict[str, Any], inspector: Any
) -> dict[str, Any]:
    from backend.src.storage.models import SessionORM

    if "messages" not in inspector.unloaded and hasattr(session_orm, "messages"):
        data["messages"] = []

        for message in cast(SessionORM, session_orm).messages:
            msg_inspector = inspect(message)
            msg_data = msg_inspector.dict.copy()

            if "message_metadata" in msg_data:
                msg_data["metadata"] = msg_data.pop("message_metadata")

            data["messages"].append(msg_data)

    return data


def _map_session_orm_to_domain(data: dict[str, Any]) -> dict[str, Any]:
    from backend.src.domain.schemas.session import (
        SessionMetadata,
        SessionRevert,
        SessionTime,
    )

    created = data.pop("created_at", None)
    updated = data.pop("updated_at", None)
    created_ts = (
        int(created.timestamp()) if isinstance(created, datetime) else (created or 0)
    )
    updated_ts = (
        int(updated.timestamp())
        if isinstance(updated, datetime)
        else (updated or created_ts or 0)
    )
    data["time"] = SessionTime(created=created_ts, updated=updated_ts)

    if "meta" in data:
        meta = data.pop("meta")
        data["metadata"] = SessionMetadata(**meta) if meta else None

    if "revert" in data:
        revert = data.pop("revert")
        data["revert"] = SessionRevert(**revert) if revert else None

    return data


def domain_to_response(
    domain_obj: T_Msgspec, response_cls: type[T_Pydantic]
) -> T_Pydantic:
    # logger.debug(f"Converting {type(domain_obj).__name__}")
    # logger.debug(f"Struct fields: {domain_obj.__struct_fields__}")

    try:
        data = msgspec.to_builtins(domain_obj)
        # logger.debug(f"Converted to builtins: {type(data)}")
    except Exception as e:
        logger.error("Failed on field inspection:")
        for field_name in domain_obj.__struct_fields__:
            field_val = getattr(domain_obj, field_name)
            logger.error("  {}: {}", field_name, type(field_val))
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

    def to_orm(self, orm_class: type[T_ORM]) -> T_ORM:
        return domain_to_orm(self, orm_class)

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


def ulid_factory() -> str:
    return str(ULID())


def utc_now() -> datetime:
    return datetime.now(UTC)
