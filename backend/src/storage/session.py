from __future__ import annotations

import base64
from typing import Any, cast, cast

import msgspec
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.src.api.schemas.session_api_schemas import (
    MessageUpdateRequest,
    NewSessionRequest,
    SessionDeleteRequest,
    SessionUpdateRequest,
)
from backend.src.domain.schemas.session import Session, SessionFull, Message, NewSession
from backend.src.domain.schemas.doc import MDNFile
from backend.src.storage.models import SessionORM, MessageORM
from backend.src.storage.utils.converters import orm_to_domain


class SessionError(Exception):
    pass


def file_part_to_dict(file_part: MDNFile) -> dict[str, Any]:
    d = msgspec.structs.asdict(file_part)
    if d.get("bytes"):
        d["bytes"] = base64.b64encode(d["bytes"]).decode("utf-8")
    return d


def dict_to_file_part(d: dict[str, Any]) -> MDNFile:
    if d.get("bytes"):
        d["bytes"] = base64.b64decode(d["bytes"])
    return msgspec.convert(d, MDNFile)


class SessionRepo:
    @staticmethod
    async def _get_session_orm(
        session: AsyncSession,
        session_id: str,
        user_id: str,
    ) -> SessionORM | None:
        stmt = (
            select(SessionORM)
            .options(
                selectinload(SessionORM.messages),
            )
            .where(SessionORM.session_id == session_id)
            .where(SessionORM.user_id == user_id)
        )

        chat = await session.execute(statement=stmt)

        return chat.scalar_one_or_none()

    @staticmethod
    def _raise_if_not_found(
        session_id: str,
        user_id: str,
        entity: str = "Conversation",
    ) -> None:
        raise SessionError(
            f"{entity} not found or access denied: session_id={session_id}, user_id={user_id}"
        )

    @staticmethod
    async def _get_message_orm(
        session: AsyncSession,
        message_id: str,
    ) -> MessageORM:
        stmt = select(MessageORM).where(MessageORM.message_id == message_id)
        result = await session.execute(stmt)
        message = result.scalar_one_or_none()
        if not message:
            raise ValueError(f"Message not found: {message_id}")
        return message

    @staticmethod
    async def create(
        session: AsyncSession, new_session: NewSessionRequest
    ) -> NewSession:
        try:
            logger.info("Creating session for user {}", new_session.user_id)

            db_chat: SessionORM = SessionORM(
                user_id=new_session.user_id,
                workspace_id=getattr(new_session, "workspace_id", "default"),
                title=new_session.title or "New Session",
            )

            session.add(db_chat)

            await session.commit()

            await session.refresh(db_chat)

            logger.success("Created session {}", db_chat.session_id)

            return orm_to_domain(db_chat, NewSession)

        except Exception as e:
            await session.rollback()
            logger.opt(exception=True).error("Error creating session")
            raise RuntimeError(f"Failed to create conversation: {e}")

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> SessionFull:
        try:
            logger.info(
                "Retrieving session: user_id={}, session_id={}", user_id, session_id
            )
            db_chat = await SessionRepo._get_session_orm(
                session,
                session_id,
                user_id=user_id,
            )

            if db_chat is None:
                SessionRepo._raise_if_not_found(session_id, user_id)

            return orm_to_domain(cast(SessionORM, db_chat), SessionFull)

        except SessionError:
            raise
        except Exception as e:
            logger.opt(exception=True).error("Error retrieving session {}", session_id)
            raise SessionError(f"Failed to retrieve conversation: {e}")

    @staticmethod
    async def get_all_by_user_id(
        session: AsyncSession,
        user_id: str,
    ) -> list[Session]:
        skip: int = 0
        limit: int = 999
        try:
            result = await session.execute(
                select(SessionORM)
                .where(SessionORM.user_id == user_id)
                .order_by(SessionORM.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            logger.debug("Session list result: {}", result)

            conversations = result.scalars().all()

            return [orm_to_domain(c, Session) for c in conversations]

        except Exception as e:
            logger.opt(exception=True).error("Error listing conversations")
            raise RuntimeError(f"Failed to list conversations: {e}")

    @staticmethod
    async def update(
        session: AsyncSession,
        session_update: SessionUpdateRequest,
    ) -> Session:
        try:
            db_chat = await SessionRepo._get_session_orm(
                session,
                session_update.session_id,
                session_update.user_id,
            )

            if db_chat is None:
                SessionRepo._raise_if_not_found(
                    session_update.session_id, session_update.user_id
                )

            chat: SessionORM = cast(SessionORM, db_chat)
            _UPDATE_ALLOWLIST = {"title", "archived", "meta"}
            update_data: dict[str, Any] = session_update.model_dump(exclude_unset=True)

            for key, value in update_data.items():
                if key in _UPDATE_ALLOWLIST and hasattr(chat, key):
                    setattr(chat, key, value)

            await session.commit()
            await session.refresh(chat)

            logger.success("Updated session {}", session_update.session_id)

            return orm_to_domain(chat, Session)

        except SessionError:
            raise
        except Exception as e:
            await session.rollback()
            logger.opt(exception=True).error("Error updating session")
            raise SessionError(f"Failed to update conversation: {e}")

    @staticmethod
    async def delete(
        session: AsyncSession,
        session_request: SessionDeleteRequest,
    ) -> bool:
        try:
            db_chat = await SessionRepo._get_session_orm(
                session,
                session_request.session_id,
                session_request.user_id,
            )

            if db_chat is None:
                SessionRepo._raise_if_not_found(
                    session_request.session_id, session_request.user_id
                )

            await session.delete(db_chat)
            await session.commit()

            logger.info("Deleted session {}", session_request.session_id)
            return True

        except SessionError:
            raise
        except Exception as e:
            await session.rollback()
            logger.opt(exception=True).error("Error deleting session")
            raise SessionError(f"Failed to delete conversation: {e}")

    @staticmethod
    async def add_message(
        session: AsyncSession,
        new_message: Message,
    ) -> MessageORM:
        try:
            if not new_message.user_id:
                raise ValueError("User ID is required to add a message")

            result = await session.execute(
                select(SessionORM).where(
                    SessionORM.session_id == new_message.session_id,
                    SessionORM.user_id == new_message.user_id,
                )
            )

            session_obj = result.unique().scalar_one_or_none()

            if not session_obj:
                raise ValueError(
                    f"Session not found or access denied: session_id={new_message.session_id}, user_id={new_message.user_id}"
                )

            parent = None
            computed_root_id = new_message.root_id
            if new_message.parent_id:
                parent = await SessionRepo._get_message_orm(
                    session, new_message.parent_id
                )
                if parent.session_id != new_message.session_id:
                    raise ValueError("Parent message not in same session")
                computed_root_id = (
                    parent.root_id if parent.root_id else parent.message_id
                )
            orm_kwargs: dict[str, Any] = dict(
                user_id=new_message.user_id,
                text=new_message.text,
                session_id=new_message.session_id,
                role=new_message.role,
                parent_id=new_message.parent_id,
                root_id=computed_root_id,
                attached_files=None,
                attached_sources=None,
            )
            # preserve caller-supplied id so SSE and DB stay in sync
            if new_message.message_id and new_message.message_id != "placeholder":
                orm_kwargs["message_id"] = new_message.message_id

            db_message = MessageORM(**orm_kwargs)

            if new_message.attached_files:
                db_message.attached_files = [
                    file_part_to_dict(f) for f in new_message.attached_files
                ]
            elif new_message.attached_sources:
                db_message.attached_sources = new_message.attached_sources

            session.add(db_message)

            await session.commit()
            await session.refresh(db_message)

            logger.debug(
                "Added {} message to {}", new_message.role, new_message.session_id
            )
            return db_message

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.opt(exception=True).error("Error adding message")
            raise RuntimeError(f"Failed to add message: {e}")

    @staticmethod
    async def update_message(
        session: AsyncSession,
        message_update: MessageUpdateRequest,
    ) -> Message:
        try:
            if not message_update.user_id:
                raise ValueError("User ID is required to update a message")

            vld_result = await session.execute(
                select(SessionORM)
                .options(selectinload(SessionORM.messages))
                .where(
                    SessionORM.session_id == message_update.session_id,
                    SessionORM.user_id == message_update.user_id,
                )
            )
            chat_valid = vld_result.scalar_one_or_none()

            if not chat_valid:
                raise ValueError(
                    f"Session not found or access denied: session_id={message_update.session_id}"
                )

            vld_result = await session.execute(
                select(MessageORM).where(
                    MessageORM.message_id == message_update.message_id
                )
            )

            existing_msg = vld_result.scalar_one_or_none()

            if not existing_msg:
                raise ValueError(
                    "Message Requested for Update didn't exist or can't be found"
                )

            if message_update.text is not None:
                existing_msg.text = message_update.text

            if existing_msg.metadata and not (
                existing_msg.message_metadata == message_update.message_metadata
            ):
                existing_msg.message_metadata = message_update.message_metadata

            session.add(existing_msg)
            await session.commit()
            await session.refresh(existing_msg)

            logger.debug(
                "Updated {} message in {}",
                message_update.role,
                message_update.session_id,
            )
            return orm_to_domain(existing_msg, Message)

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.opt(exception=True).error("Error updating message")
            raise RuntimeError(f"Failed to update message: {e}")
