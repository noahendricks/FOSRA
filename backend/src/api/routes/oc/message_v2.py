from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import get_current_user_id, get_db_session
from backend.src.api.schemas.tui_schemas import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
)
from backend.src.api.schemas.message_schemas import (
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    MessageWithParts,
    Part,
    TextPart,
    TextPartTime,
    ToolPart,
    UserMessage,
    UserMessageModel,
    UserMessageTime,
)
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.storage.models import SessionORM, MessageV2ORM, PartORM
from backend.src.storage.repos.part_repo import PartRepository
from backend.src.storage.utils.converters import ulid_factory

router = APIRouter(prefix="/oc/session", tags=["Message Operations V2"])
event_emitter = get_event_emitter()


async def _get_session_for_user(
    session: AsyncSession,
    user_id: str,
    session_id: str,
) -> SessionORM:
    stmt = (
        select(SessionORM)
        .where(SessionORM.session_id == session_id)
        .where(SessionORM.user_id == user_id)
    )
    result = await session.execute(stmt)
    session_obj = result.scalar_one_or_none()
    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_obj


async def _get_message_v2(
    session: AsyncSession,
    message_id: str,
    session_id: str,
) -> MessageV2ORM:
    stmt = (
        select(MessageV2ORM)
        .where(MessageV2ORM.message_id == message_id)
        .where(MessageV2ORM.session_id == session_id)
    )
    result = await session.execute(stmt)
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    return msg


def _message_v2_orm_to_tui(msg: MessageV2ORM, session_id: str) -> dict[str, Any]:
    msg_id = msg.message_id or ulid_factory()
    created = int(msg.created_at.timestamp()) if msg.created_at else int(time.time())

    if msg.role == "user":
        info = UserMessage(
            id=msg_id,
            sessionID=session_id,
            role="user",
            time=UserMessageTime(created=created),
            agent="fosra",
            model=UserMessageModel(
                providerID=DEFAULT_PROVIDER_ID,
                modelID=DEFAULT_MODEL_ID,
            ),
        )
    else:
        info = AssistantMessage(
            id=msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=created, completed=created),
            parentID=msg.parent_id or "",
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent="fosra",
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=1.0,
            tokens=AssistantMessageTokens(
                input=1,
                output=1,
                reasoning=1,
                cache=AssistantMessageTokensCache(read=1, write=0),
            ),
            finish="stop",
        )

    return {"info": info}


def _part_orm_to_tui(part: PartORM, session_id: str, message_id: str) -> Part:
    part_id = part.part_id or ulid_factory()
    created = int(part.created_at.timestamp()) if part.created_at else int(time.time())

    part_type = part.part_type
    data = part.data or {}

    if part_type == "text":
        return TextPart(
            id=part_id,
            sessionID=session_id,
            messageID=message_id,
            type="text",
            text=data.get("text", ""),
            synthetic=data.get("synthetic"),
            ignored=data.get("ignored"),
            time=TextPartTime(
                start=data.get("time", {}).get("start", created),
                end=data.get("time", {}).get("end"),
            ),
            metadata=data.get("metadata"),
        )
    elif part_type == "tool":
        return ToolPart(
            id=part_id,
            sessionID=session_id,
            messageID=message_id,
            type="tool",
            callID=data.get("callID", ulid_factory()),
            tool=data.get("tool", "unknown"),
            state=data.get("state", {}),
            metadata=data.get("metadata"),
        )
    else:
        return TextPart(
            id=part_id,
            sessionID=session_id,
            messageID=message_id,
            type="text",
            text=data.get("text", ""),
            metadata=data.get("metadata"),
        )


@router.get("/{session_id}/message/{message_id}")
async def get_message(
    session_id: str,
    message_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    _ = await _get_session_for_user(session, user_id, session_id)
    msg = await _get_message_v2(session, message_id, session_id)
    msg_dict = _message_v2_orm_to_tui(msg, session_id)

    part_repo = PartRepository(session)
    parts_orm = await part_repo.get_by_message(message_id)
    parts = [_part_orm_to_tui(p, session_id, message_id) for p in parts_orm]

    return MessageWithParts(info=msg_dict["info"], parts=parts)


@router.get("/{session_id}/messages")
async def list_messages(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = 50,
    before: str | None = None,
):
    _ = await _get_session_for_user(session, user_id, session_id)

    stmt = (
        select(MessageV2ORM)
        .where(MessageV2ORM.session_id == session_id)
        .order_by(MessageV2ORM.created_at.desc())
        .limit(limit)
    )
    if before:
        before_msg = await _get_message_v2(session, before, session_id)
        stmt = stmt.where(MessageV2ORM.created_at < before_msg.created_at)

    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    part_repo = PartRepository(session)
    response = []
    for msg in messages:
        msg_dict = _message_v2_orm_to_tui(msg, session_id)
        parts_orm = await part_repo.get_by_message(msg.message_id)
        parts = [_part_orm_to_tui(p, session_id, msg.message_id) for p in parts_orm]
        response.append(MessageWithParts(info=msg_dict["info"], parts=parts))

    return response


@router.post("/{session_id}/message")
async def upsert_message(
    session_id: str,
    body: dict[str, Any],
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    _ = await _get_session_for_user(session, user_id, session_id)

    message_id = body.get("messageID") or ulid_factory()
    role = body.get("role", "assistant")
    text = body.get("text", "")
    parent_id = body.get("parentID")
    root_id = body.get("rootID")
    metadata = body.get("metadata")

    stmt = select(MessageV2ORM).where(MessageV2ORM.message_id == message_id)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.text = text
        existing.role = role
        if metadata:
            existing.message_metadata = metadata
    else:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        new_msg = MessageV2ORM(
            message_id=message_id,
            session_id=session_id,
            user_id=user_id,
            parent_id=parent_id,
            root_id=root_id,
            role=role,
            text=text,
            created_at=now,
            message_metadata=metadata,
        )
        session.add(new_msg)

    await session.commit()

    msg = await _get_message_v2(session, message_id, session_id)
    msg_dict = _message_v2_orm_to_tui(msg, session_id)

    return MessageWithParts(info=msg_dict["info"], parts=[])


@router.post("/{session_id}/message/{message_id}/part")
async def upsert_part(
    session_id: str,
    message_id: str,
    body: dict[str, Any],
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    _ = await _get_session_for_user(session, user_id, session_id)
    _ = await _get_message_v2(session, message_id, session_id)

    part_id = body.get("partID") or ulid_factory()
    part_type = body.get("type", "text")
    data = body.get("data", {})

    part_repo = PartRepository(session)
    part = await part_repo.upsert(
        part_id=part_id,
        message_id=message_id,
        session_id=session_id,
        part_type=part_type,
        data=data,
    )

    part_dict = _part_orm_to_tui(part, session_id, message_id)
    await event_emitter.emit_message_part_updated(part_dict.model_dump())

    return part_dict


@router.patch("/{session_id}/message/{message_id}/part/{part_id}")
async def update_part(
    session_id: str,
    message_id: str,
    part_id: str,
    body: dict[str, Any],
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    _ = await _get_session_for_user(session, user_id, session_id)
    _ = await _get_message_v2(session, message_id, session_id)

    part_repo = PartRepository(session)
    existing = await part_repo.get(part_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Part not found")

    if "data" in body:
        updated = await part_repo.update_data(part_id, body["data"])
    elif "text" in body:
        updated = await part_repo.update_data(part_id, {"text": body["text"]})
    else:
        updated = existing

    if not updated:
        raise HTTPException(status_code=404, detail="Part not found")

    part_dict = _part_orm_to_tui(updated, session_id, message_id)
    await event_emitter.emit_message_part_updated(part_dict.model_dump())

    return part_dict


@router.delete("/{session_id}/message/{message_id}/part/{part_id}")
async def delete_part(
    session_id: str,
    message_id: str,
    part_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    _ = await _get_session_for_user(session, user_id, session_id)
    _ = await _get_message_v2(session, message_id, session_id)

    part_repo = PartRepository(session)
    deleted = await part_repo.delete(part_id)

    if deleted:
        await event_emitter.emit_message_part_removed(session_id, message_id, part_id)

    return {"ok": deleted}
