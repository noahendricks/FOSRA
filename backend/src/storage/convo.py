from __future__ import annotations

import base64
from typing import Any

import msgspec
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.src.api.schemas import MessageUpdateRequest, NewConvoRequest
from backend.src.api.schemas.api_schemas import ConvoDeleteRequest, ConvoUpdateRequest
from backend.src.domain.schemas.convo import Convo, Message, NewConvo
from backend.src.domain.schemas.doc import MDNFile
from backend.src.storage.models import ConvoORM, MessageORM
from backend.src.storage.utils.converters import orm_to_domain


def file_part_to_dict(file_part: MDNFile) -> dict[str, Any]:
    d = msgspec.structs.asdict(file_part)
    if d.get("bytes"):
        d["bytes"] = base64.b64encode(d["bytes"]).decode("utf-8")
    return d


def dict_to_file_part(d: dict[str, Any]) -> MDNFile:
    if d.get("bytes"):
        d["bytes"] = base64.b64decode(d["bytes"])
    return msgspec.convert(d, MDNFile)


class ConvoRepo:
    @staticmethod
    async def _get_convo_orm(
        session: AsyncSession,
        convo_id: str,
        user_id: str,
    ) -> ConvoORM:
        stmt = (
            select(ConvoORM)
            .options(
                selectinload(ConvoORM.messages),
            )
            .where(ConvoORM.convo_id == convo_id)
            .where(ConvoORM.user_id == user_id)
        )

        chat = await session.execute(statement=stmt)

        chat = chat.scalar_one()

        if not chat:
            raise ValueError(
                f"Conversation not found or access denied: convo_id={convo_id}, user_id={user_id}"
            )

        return chat

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
    async def create(session: AsyncSession, new_convo: NewConvoRequest) -> NewConvo:
        try:
            logger.info(f"Creating conversation for user {new_convo.user_id}")

            db_chat: ConvoORM = ConvoORM(
                user_id=new_convo.user_id,
                title=new_convo.title or "New Conversation",
            )

            session.add(db_chat)

            await session.commit()

            await session.refresh(db_chat)

            logger.success(f"Created conversation {db_chat.convo_id}")

            return orm_to_domain(db_chat, NewConvo)

        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating conversation: {e}")
            raise RuntimeError(f"Failed to create conversation: {e}")

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        user_id: str,
        convo_id: str,
    ) -> Convo:
        try:
            logger.info(
                f"Retrieving conversation: user_id={user_id}, convo_id={convo_id}"
            )
            db_chat: ConvoORM = await ConvoRepo._get_convo_orm(
                session,
                convo_id,
                user_id=user_id,
            )

            return orm_to_domain(db_chat, Convo)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving conversation {convo_id}: {e}")
            raise ValueError(f"Failed to retrieve conversation: {e}")

    @staticmethod
    async def get_all_by_user_id(
        session: AsyncSession,
        user_id: str,
    ) -> list[Convo]:
        skip: int = 0
        limit: int = 999
        try:
            result = await session.execute(
                select(ConvoORM)
                .where(ConvoORM.user_id == user_id)
                .order_by(ConvoORM.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            logger.debug("Convo list result: {}", result)

            conversations = result.scalars().all()

            return [orm_to_domain(c, Convo) for c in conversations]

        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            raise RuntimeError(f"Failed to list conversations: {e}")

    @staticmethod
    async def update(
        session: AsyncSession,
        convo_update: ConvoUpdateRequest,
    ) -> Convo:
        try:
            db_chat: ConvoORM = await ConvoRepo._get_convo_orm(
                session,
                convo_update.convo_id,
                convo_update.user_id,
            )

            update_data: dict[str, Any] = convo_update.model_dump(exclude_unset=True)

            for key, value in update_data.items():
                if hasattr(db_chat, key):
                    setattr(db_chat, key, value)

            await session.commit()
            await session.refresh(db_chat)

            logger.success(f"Updated conversation {convo_update.convo_id}")

            return orm_to_domain(db_chat, Convo)

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating conversation: {e}")
            raise RuntimeError(f"Failed to update conversation: {e}")

    @staticmethod
    async def delete(
        session: AsyncSession,
        convo_request: ConvoDeleteRequest,
    ) -> bool:
        try:
            db_chat: ConvoORM = await ConvoRepo._get_convo_orm(
                session,
                convo_request.convo_id,
                convo_request.user_id,
            )

            await session.delete(db_chat)
            await session.commit()

            logger.info(f"Deleted conversation {convo_request.convo_id}")
            return True

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error deleting conversation: {e}")
            raise RuntimeError(f"Failed to delete conversation: {e}")

    @staticmethod
    async def add_message(
        session: AsyncSession,
        new_message: Message,
    ) -> MessageORM:
        try:
            if not new_message.user_id:
                raise ValueError("User ID is required to add a message")

            result = await session.execute(
                select(ConvoORM).where(
                    ConvoORM.convo_id == new_message.convo_id,
                    ConvoORM.user_id == new_message.user_id,
                )
            )

            convo = result.unique().scalar_one_or_none()

            if not convo:
                raise ValueError(
                    f"Conversation not found or access denied: convo_id={new_message.convo_id}, user_id={new_message.user_id}"
                )

            parent = None
            computed_root_id = new_message.root_id
            if new_message.parent_id:
                parent = await ConvoRepo._get_message_orm(
                    session, new_message.parent_id
                )
                if parent.convo_id != new_message.convo_id:
                    raise ValueError("Parent message not in same conversation")
                computed_root_id = (
                    parent.root_id if parent.root_id else parent.message_id
                )
            db_message = MessageORM(
                user_id=new_message.user_id,
                text=new_message.text,
                convo_id=new_message.convo_id,
                role=new_message.role,
                parent_id=new_message.parent_id,
                root_id=computed_root_id,
                attached_files=None,
                attached_sources=None,
            )

            if new_message.attached_files:
                db_message.attached_files = [
                    file_part_to_dict(f) for f in new_message.attached_files
                ]
            elif new_message.attached_sources:
                db_message.attached_sources = new_message.attached_sources

            session.add(db_message)

            await session.commit()
            await session.refresh(db_message)

            logger.debug(f"Added {new_message.role} message to {new_message.convo_id}")
            return db_message

        except ValueError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error adding message: {e}")
            raise RuntimeError(f"Failed to add message: {e}")


async def update_message(
    session: AsyncSession,
    message_update: MessageUpdateRequest,
) -> Message:
    try:
        if not message_update.user_id:
            raise ValueError("User ID is required to update a message")

        vld_result = await session.execute(
            select(ConvoORM)
            .options(selectinload(ConvoORM.messages))
            .where(
                ConvoORM.convo_id == message_update.convo_id,
                ConvoORM.user_id == message_update.user_id,
            )
        )
        chat_valid = vld_result.scalar_one_or_none()

        if not chat_valid:
            raise ValueError(
                f"Conversation not found or access denied: convo_id={message_update.convo_id}"
            )

        vld_result = await session.execute(
            select(MessageORM).where(MessageORM.message_id == message_update.message_id)
        )

        existing_msg = vld_result.scalar_one_or_none()

        if not existing_msg:
            raise ValueError(
                "Message Requested for Update didn't exist or can't be found"
            )

        if message_update.text is not None:
            existing_msg.text = message_update.text

        if (
            existing_msg.metadata
            and not existing_msg.message_metadata == message_update.message_metadata
        ):
            existing_msg.message_metadata = message_update.message_metadata

        session.add(existing_msg)
        await session.commit()
        await session.refresh(existing_msg)

        logger.debug(
            f"Updated {message_update.role} message in {message_update.convo_id}"
        )
        return orm_to_domain(existing_msg, Message)

    except ValueError:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error updating message: {e}")
        raise RuntimeError(f"Failed to update message: {e}")
