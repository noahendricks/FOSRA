from __future__ import annotations
import base64
import msgspec
from typing import Any

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas import (
    ConvoRequest,
    MessageUpdateRequest,
    NewConvoRequest,
)
from backend.src.api.schemas.api_schemas import (
    MessageRequest,
    ConvoDeleteRequest,
    ConvoUpdateRequest,
)
from backend.src.domain.exceptions.exceptions import (
    ConvoRetrievalError,
    ConvoStorageError,
    TenantContextError,
)
from backend.src.domain.schemas.schemas import FilePartDomain
from backend.src.storage.models import (
    ConvoORM,
    MessageORM,
)

from backend.src.storage.utils.converters import orm_to_domain
from backend.src.domain.schemas import (
    ConvoFull,
    NewConvo,
    Message,
    RetrievedResult,
)


from backend.src.domain.enums import MessageRole


def file_part_to_dict(file_part: FilePartDomain) -> dict:
    d = msgspec.structs.asdict(file_part)
    if d.get("bytes"):
        d["bytes"] = base64.b64encode(d["bytes"]).decode("utf-8")
    return d


def dict_to_file_part(d: dict) -> FilePartDomain:
    if d.get("bytes"):
        d["bytes"] = base64.b64decode(d["bytes"])
    return msgspec.convert(d, FilePartDomain)


class ConvoRepo:
    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    async def _get_convo_orm(
        session: AsyncSession,
        convo_id: str,
        user_id: str,
        workspace_id: str = "",
    ) -> ConvoORM:
        stmt = (
            select(ConvoORM)
            .options(
                selectinload(ConvoORM.messages),  # Eager load messages
                selectinload(ConvoORM.user),  # Eager load user
                selectinload(ConvoORM.workspace),  # Eager load workspace
            )
            .where(ConvoORM.convo_id == convo_id)
            .where(ConvoORM.user_id == user_id)
        )

        # logger.info(f"after orm get: {vars(result)}")

        chat = await session.execute(statement=stmt)

        chat = chat.scalar_one()

        if not chat:
            raise ConvoRetrievalError(
                convo_id=convo_id,
                user_id=user_id,
                reason="Conversation not found or access denied.",
                remediation="Verify conversation ID and user permissions.",
            )

        # logger.info(f"after orm to chat: {vars(chat)}")

        return chat

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    @staticmethod
    async def create(session: AsyncSession, new_convo: NewConvoRequest) -> NewConvo:
        try:
            logger.info(
                f"Right before upsert to db: User ID: {
                    new_convo.user_id
                }, workspace_id: {new_convo.workspace_id} "
            )

            db_chat: ConvoORM = ConvoORM(
                user_id=new_convo.user_id,
                workspace_id=new_convo.workspace_id,
                title=new_convo.title or "New Conversation",
            )

            logger.info(
                f"atfer upsert to db: User ID: {db_chat.user_id}, workspace_id: {
                    db_chat.workspace_id
                } "
            )

            session.add(db_chat)

            await session.commit()

            await session.refresh(db_chat)

            logger.success(f"Created conversation {db_chat.convo_id}")

            return orm_to_domain(db_chat, NewConvo)

        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating conversation: {e}")
            raise ConvoStorageError(
                operation="create",
                convo_id="N/A",
                reason=str(e),
                remediation="Ensure input data is valid and session is active.",
            )

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        user_id: str,
        convo_id: str,
    ) -> ConvoFull:
        try:
            logger.info(
                f"user id and convo_id get_by_id entrance: user_id: {user_id}, convo_id: {convo_id}"
            )
            db_chat: ConvoORM = await ConvoRepo._get_convo_orm(
                session,
                convo_id,
                user_id=user_id,
            )
            logger.info(
                f"user id and convo_id get_by_id exit: user_id: {user_id}, convo_id: {convo_id}"
            )

            # WARN: May be causing issues
            # db_chat.metadata = None

            # logger.info(f"db_chat vars: {vars(db_chat)}")

            return orm_to_domain(db_chat, ConvoFull)

        except ConvoRetrievalError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving conversation {convo_id}: {e}")
            raise ConvoRetrievalError(
                convo_id=convo_id,
                reason=str(e),
                user_id="Ensure valid user request data and Database is reachable.",
                remediation="Check database connection.",
            ) from e

    @staticmethod
    async def get_all_by_workspace_id(
        session: AsyncSession,
        user_id: str,
        workspace_id: str,
    ) -> list[ConvoFull]:
        skip: int = 0
        limit: int = 999
        try:
            # print(user_id, workspace_id)
            # print(type(user_id), type(workspace_id))

            result = await session.execute(
                select(ConvoORM)
                .where(
                    ConvoORM.user_id == user_id,
                    ConvoORM.workspace_id == workspace_id,
                )
                .order_by(ConvoORM.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            print(result)
            print(vars(result))

            conversations = result.scalars().all()

            return [orm_to_domain(c, ConvoFull) for c in conversations]

        except Exception as e:
            logger.error(f"Error listing workspace conversations: {e}")
            raise ConvoRetrievalError(
                convo_id="",
                reason=str(e),
                user_id=user_id,
                remediation="Check database connection.",
            )

    @staticmethod
    async def update(
        session: AsyncSession,
        convo_update: ConvoUpdateRequest,
    ) -> ConvoFull:
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

            return orm_to_domain(db_chat, ConvoFull)

        except ConvoRetrievalError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating conversation: {e}")
            raise ConvoStorageError(
                operation="update",
                convo_id=convo_update.convo_id,
                user_id=convo_update.user_id,
                reason=str(e),
                remediation="Ensure conversation exists and update data is valid.",
            ) from e

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

        except ConvoRetrievalError:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error deleting conversation: {e}")
            raise ConvoStorageError(
                operation="delete",
                convo_id=convo_request.convo_id,
                user_id=convo_request.user_id,
                reason=str(e),
                remediation="Ensure conversation exists.",
            ) from e

    # =========================================================================
    # Message Operations
    # =========================================================================

    @staticmethod
    async def add_message(
        session: AsyncSession,
        new_message: Message,
    ) -> MessageORM:
        try:
            if not new_message.user_id:
                raise TenantContextError("User ID is required to add a message")

            # Authorization check
            result = await session.execute(
                select(ConvoORM).where(
                    ConvoORM.convo_id == new_message.convo_id,
                    ConvoORM.user_id == new_message.user_id,
                )
            )

            convo = result.unique().scalar_one_or_none()

            if not convo:
                raise ConvoRetrievalError(
                    convo_id=new_message.convo_id,
                    user_id=new_message.user_id,
                    reason="Conversation not found or access denied.",
                    remediation="Verify conversation ID and user permissions.",
                )

                # NOTE: For now parsing assistant sources to string using source id
            db_message = MessageORM(
                user_id=new_message.user_id,
                text=new_message.text,
                convo_id=new_message.convo_id,
                role=new_message.role,
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

        except (TenantContextError, ConvoRetrievalError):
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Error adding message: {e}")
            raise ConvoStorageError(
                operation="add message",
                convo_id=new_message.convo_id,
                user_id=new_message.user_id or "Unknown",
                reason=f"Failed to add message: {e}",
                remediation="Verify conversation exists and session is valid.",
            ) from e


async def update_message(
    session: AsyncSession,
    message_update: MessageUpdateRequest,
) -> Message:
    try:
        if not message_update.user_id:
            raise TenantContextError("User ID is required to add a message")

        # Authorization check
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
            raise ConvoRetrievalError(
                convo_id=message_update.convo_id,
                user_id=message_update.user_id,
                reason="Conversation not found or access denied.",
                remediation="Verify conversation ID and user permissions.",
            )

        vld_result = await session.execute(
            select(MessageORM).where(MessageORM.message_id == message_update.message_id)
        )

        existing_msg = vld_result.scalar_one_or_none()

        if not existing_msg:
            raise ValueError(
                "Message Requested for Update didn't exist or can't be found"
            )

        existing_msg.text = message_update.text

        if (
            existing_msg.metadata
            and not existing_msg.message_metadata == message_update.message_metadata
        ):
            # TODO: DB UPDATE
            existing_msg.message_metadata = message_update.message_metadata

        session.add(existing_msg)
        await session.commit()
        await session.refresh(existing_msg)

        logger.debug(
            f"Added {message_update.role} message to {message_update.convo_id}"
        )
        return orm_to_domain(existing_msg, Message)

    except (TenantContextError, ConvoRetrievalError):
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error adding message: {e}")
        raise ConvoStorageError(
            operation="add message",
            convo_id=message_update.convo_id,
            user_id=message_update.user_id or "Unknown",
            reason=f"Failed to add message: {e}",
            remediation="Verify conversation exists and session is valid.",
        ) from e
