from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec
from loguru import logger
from qdrant_client.conversions.common_types import QueryResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas.message_schemas import (
    AssistantMessage,
    AssistantMessagePath,
    AssistantMessageTime,
    AssistantMessageTokens,
    AssistantMessageTokensCache,
    UserMessage,
    UserMessageModel,
    UserMessageTime,
)
from backend.src.api.schemas.session_api_schemas import (
    NewSessionRequest,
    NewSessionResponse,
    SessionDeleteRequest,
    SessionFullResponse,
    SessionListItemResponse,
    SessionTime,
    SessionUpdateRequest,
)
from backend.src.api.schemas.tui_control_schemas import (
    TextPart,
    UIMessage,
)
from backend.src.api.schemas.tui_schemas import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    PROJECT_DIR,
)
from backend.src.domain.enums import MessageRole, VectorStoreType
from backend.src.domain.schemas.doc import MDNFile
from backend.src.domain.schemas.session import Message, NewSession, Session
from backend.src.settings import ScoredRetrieval
from backend.src.storage.session import SessionRepo
from backend.src.storage.utils.converters import domain_to_response, utc_now

if TYPE_CHECKING:
    from backend.src.storage.models import MessageORM


def _ts_to_dt(ts: int | float | datetime | None) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if ts is None:
        return utc_now()
    return datetime.fromtimestamp(ts, tz=UTC)


def _session_time_to_dts(time_obj) -> tuple[datetime, datetime]:
    created = getattr(time_obj, "created", 0) if time_obj else 0
    updated = getattr(time_obj, "updated", created) if time_obj else created
    return _ts_to_dt(created), _ts_to_dt(updated)


def _build_tui_message(msg: Message, session_id: str) -> UserMessage | AssistantMessage:
    role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
    ts = int(msg.timestamp.timestamp()) if msg.timestamp else 0
    msg_id = msg.message_id or ""

    if role_val == "user":
        return UserMessage(
            id=msg_id,
            sessionID=session_id,
            role="user",
            time=UserMessageTime(created=ts),
            agent="default",
            model=UserMessageModel(
                providerID=DEFAULT_PROVIDER_ID,
                modelID=DEFAULT_MODEL_ID,
            ),
        )
    else:
        return AssistantMessage(
            id=msg_id,
            sessionID=session_id,
            role="assistant",
            time=AssistantMessageTime(created=ts, completed=ts),
            parentID=msg.parent_id or "",
            modelID=DEFAULT_MODEL_ID,
            providerID=DEFAULT_PROVIDER_ID,
            mode="default",
            agent="default",
            path=AssistantMessagePath(cwd=PROJECT_DIR, root=PROJECT_DIR),
            cost=0,
            tokens=AssistantMessageTokens(
                input=0,
                output=0,
                reasoning=0,
                cache=AssistantMessageTokensCache(read=0, write=0),
            ),
            finish="stop",
        )


class SessionService:
    @staticmethod
    async def create_session(
        session: AsyncSession,
        new_session: NewSessionRequest,
    ) -> NewSessionResponse:
        logger.info("Creating session for user {}", new_session.user_id)

        session_obj: NewSession = await SessionRepo().create(
            session=session,
            new_session=new_session,
        )

        logger.success("Created session: {}", session_obj.session_id)

        created_at, updated_at = _session_time_to_dts(session_obj.time)

        return NewSessionResponse(
            user_id=session_obj.user_id,
            session_id=session_obj.session_id,
            title=session_obj.title,
            directory=session_obj.directory or "",
            version=session_obj.version or "1",
            parent_id=session_obj.parent_id,
            session_metadata=session_obj.session_metadata,
            time=SessionTime(
                created=int(created_at.timestamp()), updated=int(updated_at.timestamp())
            ),
        )

    @staticmethod
    async def get_session_by_id(
        session: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> SessionFullResponse:
        logger.info("Retrieving session: {}", session_id)

        logger.bind(_structured={"user_id": user_id, "session_id": session_id}).debug(
            "get_session_by_id"
        )

        session_obj: Session = await SessionRepo.get_by_id(
            session=session,
            user_id=user_id,
            session_id=session_id,
        )

        tui_messages = [
            _build_tui_message(msg, session_id) for msg in (session_obj.messages or [])
        ]

        created_at, updated_at = _session_time_to_dts(session_obj.time)

        return SessionFullResponse(
            user_id=session_obj.user_id,
            session_id=session_obj.session_id,
            title=session_obj.title or "New Session",
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(tui_messages),
            messages=tui_messages,
            directory=session_obj.directory or PROJECT_DIR,
            version=session_obj.version or "1",
            parent_id=session_obj.parent_id,
            permission=session_obj.permission,
            revert=msgspec.to_builtins(session_obj.revert)
            if session_obj.revert
            else None,
            metadata=msgspec.to_builtins(session_obj.metadata)
            if session_obj.metadata
            else None,
            time=SessionTime(
                created=int(created_at.timestamp()), updated=int(updated_at.timestamp())
            ),
        )

    @staticmethod
    async def list_sessions(
        session: AsyncSession,
        user_id: str,
    ) -> list[SessionListItemResponse]:
        logger.debug("Listing sessions for user {}", user_id)

        sessions: list[Session] = await SessionRepo().get_all_by_user_id(
            session=session,
            user_id=user_id,
        )

        logger.success("Retrieved {} sessions for user {}", len(sessions), user_id)

        results = []
        for s in sessions:
            created_at, updated_at = _session_time_to_dts(s.time)
            results.append(
                SessionListItemResponse(
                    user_id=s.user_id,
                    session_id=s.session_id,
                    title=s.title or "",
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=0,
                    directory=s.directory or PROJECT_DIR,
                    version=s.version or "1",
                    parent_id=s.parent_id,
                    permission=s.permission,
                    revert=msgspec.to_builtins(s.revert) if s.revert else None,
                    metadata=msgspec.to_builtins(s.metadata) if s.metadata else None,
                )
            )
        return results

    @staticmethod
    async def update_session(
        session: AsyncSession,
        session_update: SessionUpdateRequest,
    ) -> SessionFullResponse:
        logger.info("Updating session: {}", session_update.session_id)

        session_obj: Session = await SessionRepo.update(
            session=session,
            session_update=session_update,
        )

        logger.success("Updated session: {}", session_update.session_id)

        created_at, updated_at = _session_time_to_dts(session_obj.time)

        return SessionFullResponse(
            user_id=session_obj.user_id,
            session_id=session_obj.session_id,
            title=session_obj.title or "",
            created_at=created_at,
            updated_at=updated_at,
            message_count=0,
            directory=session_obj.directory or PROJECT_DIR,
            version=session_obj.version or "1",
            parent_id=session_obj.parent_id,
            permission=session_obj.permission,
            revert=msgspec.to_builtins(session_obj.revert)
            if session_obj.revert
            else None,
            metadata=msgspec.to_builtins(session_obj.metadata)
            if session_obj.metadata
            else None,
            time=SessionTime(
                created=int(created_at.timestamp()), updated=int(updated_at.timestamp())
            ),
        )

    @staticmethod
    async def delete_session(
        session: AsyncSession,
        session_request: SessionDeleteRequest,
    ) -> bool:
        logger.info("Deleting session: {}", session_request.session_id)

        deleted = await SessionRepo.delete(
            session=session,
            session_request=session_request,
        )

        if deleted:
            logger.success("Deleted session: {}", session_request.session_id)
        else:
            logger.warning("Session not found: {}", session_request.session_id)

        return deleted

    @staticmethod
    async def unpack_message(
        message: UIMessage, session_id: str, user_id: str
    ) -> Message:
        if message.role == "user":
            _role = MessageRole.USER

        elif message.role == "assistant":
            _role = MessageRole.ASSISTANT
        else:
            _role = MessageRole.USER

        unpacked: Message = Message(
            role=_role,
            session_id=session_id,
            text="",
            message_id=message.id,
            user_id=user_id,
            parent_id=message.message_metadata.get("parent_id")
            if message.message_metadata
            else None,
            root_id=message.message_metadata.get("root_id")
            if message.message_metadata
            else None,
            attached_sources=message.sources,
        )

        for part in message.parts:
            if isinstance(part, TextPart) and part.type == "text":
                if not unpacked.text or unpacked.text == "":
                    unpacked.text += part.text
                else:
                    unpacked.text += "\n"
                    unpacked.text += part.text
            if isinstance(part, MDNFile) and part.type == "file":
                if unpacked.attached_files:
                    unpacked.attached_files.append(part)
                else:
                    unpacked.attached_files = []
                    unpacked.attached_files.append(part)

        return unpacked

    @staticmethod
    async def save_message(
        session: AsyncSession,
        session_id: str,
        user_id: str,
        message: UIMessage,
    ) -> UserMessage | AssistantMessage:
        logger.bind(_structured={"session_id": session_id, "user_id": user_id}).info(
            "processing user message for session"
        )

        logger.bind(_structured={"ui message": message}).debug("[UI MESSAGE]")
        msg: Message = await SessionService.unpack_message(
            message, session_id=session_id, user_id=user_id
        )

        logger.bind(_structured={"backend message": msg.to_dict()}).debug(
            "[BACKEND MESSAGE]"
        )
        _: MessageORM = await SessionRepo.add_message(
            session=session,
            new_message=msg,
        )

        return _build_tui_message(msg, session_id)

    @staticmethod
    async def parse_retrievals(retrievals: QueryResponse, store_type: VectorStoreType):
        sources = []

        match store_type:
            case VectorStoreType.QDRANT:
                points = retrievals.points

                for p in points:
                    if p.payload is None:
                        continue
                    logger.debug("Payload keys: {}", p.payload.keys())
                    sources.append(
                        ScoredRetrieval(
                            score=p.score,
                            text=p.payload["chunk"],
                            chunk_id=p.payload["chunk_id"],
                            doc_title=p.payload["title"],
                            doc_id=p.payload["doc_id"],
                            page_number=p.payload["page_number"],
                            start_index=p.payload["start_index"],
                            end_index=p.payload["end_index"],
                        )
                    )
                return sources
