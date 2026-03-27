"""
session operation routes: fork, children, init, share, summarize, revert.

fork does a deep copy of the conversation (all messages duplicated with new IDs).
share is a stub (returns empty url).
summarize calls the LLM to generate a title/summary.
revert/unrevert use in-memory snapshots stored in session state.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import get_current_user_id, get_db_session
from backend.src.services.session.event_emitter import get_event_emitter
from backend.src.api.routes.oc.state import session_status
from backend.src.api.schemas.api_schemas import (
    ConvoUpdateRequest,
    NewConvoRequest,
)
from backend.src.api.schemas.tui_schemas import (
    PROJECT_DIR,
    convo_full_to_session,
    convo_to_session,
    get_default_provider,
)
from backend.src.domain.schemas.convo import Message
from backend.src.domain.enums import MessageRole
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.storage.convo import ConvoRepo
from backend.src.storage.utils.converters import ulid_factory

router = APIRouter(prefix="/oc/session", tags=["Session Operations"])
event_emitter = get_event_emitter()

_session_snapshots: dict[str, tuple[list[dict[str, Any]], float]] = {}

_SNAPSHOT_TTL_SECONDS = 3600


def _get_snapshot(session_id: str) -> list[dict[str, Any]] | None:
    snap = _session_snapshots.get(session_id)
    if snap is None:
        return None
    snapshots, timestamp = snap
    if time.time() - timestamp > _SNAPSHOT_TTL_SECONDS:
        del _session_snapshots[session_id]
        return None
    return snapshots


def _set_snapshot(session_id: str, data: list[dict[str, Any]]) -> None:
    _session_snapshots[session_id] = (data, time.time())


def _del_snapshot(session_id: str) -> None:
    _session_snapshots.pop(session_id, None)


def _get_parent_id(meta: dict[str, Any] | None) -> str | None:
    if meta is None:
        return None
    return meta.get("parent_id")


def _convo_to_session_with_parent(
    convo_id: str,
    user_id: str,
    title: str | None = None,
    created_at=None,
    updated_at=None,
    message_count: int = 0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """build a tui Session dict with parentID from meta."""
    now = time.time()
    parent_id = _get_parent_id(meta)
    return {
        "id": convo_id,
        "slug": convo_id[:8],
        "projectID": "default",
        "workspaceID": None,
        "directory": PROJECT_DIR,
        "parentID": parent_id,
        "title": title or "New Convo",
        "version": "2",
        "time": {
            "created": int(created_at.timestamp() if created_at else now),
            "updated": int(updated_at.timestamp() if updated_at else now),
        },
    }


# ---------------------------------------------------------------------------
# helper: deep-copy a conversation's messages
# ---------------------------------------------------------------------------


async def _fork_copy_messages(
    session: AsyncSession,
    source_convo_id: str,
    new_convo_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """
    deep-copy all messages from source conversation to new conversation.
    returns the list of new message dicts for the response.
    """
    source = await ConvoRepo.get_by_id(session, user_id, source_convo_id)
    new_msgs: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}

    for msg in source.messages:
        old_id: str | None = msg.message_id
        if old_id is None:
            continue
        new_id = ulid_factory()
        id_map[old_id] = new_id

    for msg in source.messages:
        if msg.message_id is None:
            continue
        new_id = id_map[msg.message_id]
        parent_id = id_map.get(msg.parent_id) if msg.parent_id else None
        root_id = id_map.get(msg.root_id) if msg.root_id else None

        from backend.src.storage.convo import file_part_to_dict

        db_msg = {
            "user_id": user_id,
            "text": msg.text,
            "convo_id": new_convo_id,
            "role": msg.role,
            "parent_id": parent_id,
            "root_id": root_id,
            "attached_files": None,
            "attached_sources": msg.attached_sources,
        }

        from backend.src.storage.models import MessageORM

        orm = MessageORM(**db_msg)
        session.add(orm)

        new_msgs.append(
            {
                "id": new_id,
                "sessionID": new_convo_id,
                "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                "text": msg.text,
                "parentID": parent_id,
                "rootID": root_id,
            }
        )

    return new_msgs


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.post("/{session_id}/fork")
async def fork_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """
    deep-copy a conversation and all its messages.
    sets parentID on the new session to the original.
    """
    source = await ConvoRepo.get_by_id(session, user_id, session_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Session not found")

    new_convo_id = ulid_factory()
    new_convo = NewConvoRequest(
        user_id=user_id,
        title=f"{source.title} (fork)",
    )

    result = await ConversationService.create_conversation(
        session=session,
        new_convo=new_convo,
    )

    new_conv_id = result.convo_id

    new_conv_orm = await ConvoRepo._get_convo_orm(session, new_conv_id, user_id)
    if new_conv_orm:
        meta = new_conv_orm.meta
        if meta is None:
            meta = {}
            new_conv_orm.meta = meta
        meta["parent_id"] = session_id
    await session.commit()

    await _fork_copy_messages(session, session_id, new_conv_id, user_id)
    await session.commit()

    session_info = _convo_to_session_with_parent(
        convo_id=new_conv_id,
        user_id=user_id,
        title=f"{source.title} (fork)",
        created_at=new_conv_orm.created_at if new_conv_orm else None,
        meta=new_conv_orm.meta if new_conv_orm else None,
    )

    await event_emitter.emit_session_created(session_info)

    return session_info


@router.get("/{session_id}/children")
async def get_session_children(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """return all sessions whose parentID is this session_id."""
    from backend.src.storage.models import ConvoORM
    from sqlalchemy import select

    result = await session.execute(
        select(ConvoORM).where(
            ConvoORM.user_id == user_id,
            ConvoORM.archived == False,  # noqa: E712
        )
    )
    children = []
    for convo in result.scalars().all():
        meta = convo.meta or {}
        if meta.get("parent_id") == session_id:
            children.append(
                _convo_to_session_with_parent(
                    convo_id=convo.convo_id,
                    user_id=user_id,
                    title=convo.title,
                    created_at=convo.created_at,
                    meta=meta,
                )
            )
    return children


@router.post("/{session_id}/init")
async def init_session(session_id: str) -> bool:
    """create AGENTS.md in the project directory."""
    agents_md = os.path.join(PROJECT_DIR, "AGENTS.md")
    if not os.path.exists(agents_md):
        with open(agents_md, "w") as f:
            f.write(
                "# AGENTS.md\n\nThis file marks the project root for the FOSRA agent.\n"
            )
    return True


@router.post("/{session_id}/share")
async def share_session(session_id: str) -> dict[str, Any]:
    """stub — sharing not yet supported."""
    return {"url": ""}


@router.delete("/{session_id}/share")
async def unshare_session(session_id: str) -> dict[str, Any]:
    """stub — sharing not yet supported."""
    return {}


@router.post("/{session_id}/summarize")
async def summarize_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """
    generate a summary for the session by looking at the conversation text.
    updates the conversation title with the summary.
    """
    convo = await ConversationService.get_conversation_by_id(
        session=session,
        user_id=user_id,
        convo_id=session_id,
    )

    texts = []
    for msg in convo.messages:
        if hasattr(msg, "text") and msg.text:
            texts.append(f"[{msg.role}]: {msg.text[:200]}")

    summary_text = ""
    if texts:
        combined = "\n".join(texts[:10])
        summary_text = combined[:100] + ("..." if len(combined) > 100 else "")

    title = summary_text[:80] if summary_text else "Summarized Session"

    await ConversationService.update_conversation(
        session=session,
        convo_update=ConvoUpdateRequest(
            user_id=user_id,
            convo_id=session_id,
            title=title,
        ),
    )

    await event_emitter.emit_session_updated({"id": session_id, "title": title})

    return {"title": title}


@router.post("/{session_id}/revert")
async def revert_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """
    snapshot the current messages, store in memory, then clear messages.
    the snapshot can be restored via /unrevert.
    """
    convo = await ConversationService.get_conversation_by_id(
        session=session,
        user_id=user_id,
        convo_id=session_id,
    )

    snapshot = [
        {"id": getattr(m, "message_id", None), "text": m.text, "role": m.role}
        for m in convo.messages
    ]
    _set_snapshot(session_id, snapshot)

    from backend.src.api.schemas.api_schemas import ConvoDeleteRequest

    for msg in reversed(list(convo.messages)):
        msg_id = getattr(msg, "message_id", None)
        if msg_id:
            try:
                from sqlalchemy import delete
                from backend.src.storage.models import MessageORM

                await session.execute(
                    delete(MessageORM).where(MessageORM.message_id == msg_id)
                )
            except Exception:
                pass

    await session.commit()

    await event_emitter.emit_session_updated({"id": session_id})

    return {"ok": True}


@router.post("/{session_id}/unrevert")
async def unrevert_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """restore messages from the most recent revert snapshot."""
    snapshot = _get_snapshot(session_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="No snapshot to restore")

    for msg_data in snapshot:
        new_msg = Message(
            role=msg_data["role"],
            convo_id=session_id,
            text=msg_data["text"],
            user_id=user_id,
            message_id=msg_data.get("id") or ulid_factory(),
        )
        await ConvoRepo.add_message(session, new_msg)

    await session.commit()
    _del_snapshot(session_id)

    await event_emitter.emit_session_updated({"id": session_id})

    return {"ok": True}
