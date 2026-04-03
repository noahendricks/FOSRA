"""
message operation routes: single message get/delete, part update/delete.

pagination on list_messages is handled as query params on the existing
route in tui.py (limit / before).
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import get_current_user_id, get_db_session
from backend.src.api.schemas.tui_schemas import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    Message,
    Part,
    TextPart,
    TextPartTime,
    ToolPart,
    ToolStateCompleted,
    ToolStateCompletedTime,
    UserMessage,
    UserMessageModel,
    UserMessageTime,
    message_to_tui,
)
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.storage.models import ConvoORM, MessageORM
from backend.src.storage.utils.converters import ulid_factory

router = APIRouter(prefix="/oc/session", tags=["Message Operations"])
event_emitter = get_event_emitter()


async def _get_message_orm(
    session: AsyncSession,
    message_id: str,
    user_id: str,
    convo_id: str,
) -> MessageORM:
    stmt = (
        select(MessageORM)
        .where(MessageORM.message_id == message_id)
        .where(MessageORM.convo_id == convo_id)
        .where(MessageORM.user_id == user_id)
    )
    result = await session.execute(stmt)
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    return msg


async def _get_convo_for_user(
    session: AsyncSession,
    user_id: str,
    convo_id: str,
) -> ConvoORM:
    stmt = (
        select(ConvoORM)
        .where(ConvoORM.convo_id == convo_id)
        .where(ConvoORM.user_id == user_id)
    )
    result = await session.execute(stmt)
    convo = result.scalar_one_or_none()
    if not convo:
        raise HTTPException(status_code=404, detail="Session not found")
    return convo


def _message_orm_to_tui(msg: MessageORM, session_id: str) -> dict[str, Any]:
    """convert a MessageORM row to a TUI message dict with parts."""
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

    parts: list[Part] = []

    if msg.text:
        parts.append(
            TextPart(
                id=ulid_factory(),
                sessionID=session_id,
                messageID=msg_id,
                type="text",
                text=msg.text,
                time=TextPartTime(start=created, end=created),
            )
        )

    if msg.attached_sources:
        for source in msg.attached_sources:
            parts.append(
                ToolPart(
                    id=ulid_factory(),
                    sessionID=session_id,
                    messageID=msg_id,
                    type="tool",
                    callID=ulid_factory(),
                    tool="search_knowledge_base",
                    state=ToolStateCompleted(
                        status="completed",
                        input={"query": "retrieval"},
                        output=str(source),
                        title="Knowledge Base Search",
                        metadata=source if isinstance(source, dict) else {},
                        time=ToolStateCompletedTime(start=created, end=created),
                    ),
                )
            )

    return {"info": info, "parts": parts}


@router.get("/{session_id}/message/{message_id}")
async def get_message(
    session_id: str,
    message_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """return a single message with its parts."""
    await _get_convo_for_user(session, user_id, session_id)
    msg = await _get_message_orm(session, message_id, user_id, session_id)
    return _message_orm_to_tui(msg, session_id)


@router.delete("/{session_id}/message/{message_id}")
async def delete_message(
    session_id: str,
    message_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """delete a message and publish message.removed event."""
    await _get_convo_for_user(session, user_id, session_id)
    msg = await _get_message_orm(session, message_id, user_id, session_id)

    await session.execute(delete(MessageORM).where(MessageORM.message_id == message_id))
    await session.commit()

    await event_emitter.emit_message_removed(session_id, message_id)
    return True


@router.patch("/{session_id}/message/{message_id}/part/{part_id}")
async def update_part(
    session_id: str,
    message_id: str,
    part_id: str,
    body: dict[str, Any],
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """
    update part content (text field).
    publishes message.part.updated with the updated TextPart.
    """
    await _get_convo_for_user(session, user_id, session_id)
    msg = await _get_message_orm(session, message_id, user_id, session_id)

    if "text" in body:
        msg.text = body["text"]
        await session.commit()

    created = int(msg.created_at.timestamp()) if msg.created_at else int(time.time())
    updated_part = TextPart(
        id=part_id,
        sessionID=session_id,
        messageID=message_id,
        type="text",
        text=body.get("text", msg.text or ""),
        time=TextPartTime(start=created, end=int(time.time())),
    )

    await event_emitter.emit_message_part_updated(updated_part)
    return {"ok": True}


@router.delete("/{session_id}/message/{message_id}/part/{part_id}")
async def delete_part(
    session_id: str,
    message_id: str,
    part_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """
    stub — parts are embedded in messages so we can't delete a single part
    without restructuring. publish part.removed event for the tui to handle.
    """
    await _get_convo_for_user(session, user_id, session_id)

    await event_emitter.emit_message_part_removed(session_id, message_id, part_id)
    return True
