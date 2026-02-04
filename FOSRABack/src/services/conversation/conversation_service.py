from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from FOSRABack.src.api.schemas import (
    MessageResponse,
    NewConvoRequest,
)
from FOSRABack.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoListItemResponse,
    FilePart,
    NewConvoResponse,
    TextPart,
    UIMessage,
)
from FOSRABack.src.domain.enums import MessageRole
from FOSRABack.src.domain.schemas import (
    ConvoFull,
    Message,
    NewConvo,
)
from FOSRABack.src.domain.schemas.schemas import FilePartDomain
from FOSRABack.src.storage.utils.converters import (
    domain_to_response,
    orm_to_domain,
    pydantic_to_domain,
)
from FOSRABack.src.storage.repos.convo_repo import ConvoRepo


from FOSRABack.src.api.schemas import (
    ConvoFullResponse,
    ConvoUpdateRequest,
)

# pyright: strict
if TYPE_CHECKING:
    from FOSRABack.src.storage.models import MessageORM


class ConversationService:
    # =========================================================================
    # Conversation Lifecycle
    # =========================================================================

    @staticmethod
    async def create_conversation(
        session: AsyncSession,
        new_convo: NewConvoRequest,
    ) -> NewConvoResponse:
        logger.info(
            f"Creating conversation for user {new_convo.user_id} "
            f"in workspace {new_convo.workspace_id}"
        )

        try:
            conversation: NewConvo = await ConvoRepo.create(
                session=session,
                new_convo=new_convo,
            )

            logger.success(f"Created conversation: {conversation.convo_id}")

            return domain_to_response(
                conversation,
                NewConvoResponse,
            )

        except Exception as e:
            raise e

    @staticmethod
    async def get_conversation_by_id(
        session: AsyncSession,
        user_id: str,
        convo_id: str,
    ) -> ConvoFullResponse:
        logger.info(f"Retrieving conversation: {convo_id}")

        try:
            logger.info(
                f"user id and convo_id get_conversation_by_id entrance : user_id: {user_id}, convo_id: {convo_id}"
            )

            conversation: ConvoFull = await ConvoRepo.get_by_id(
                session=session,
                user_id=user_id,
                convo_id=convo_id,
            )

            # logger.info(
            #     f"Convo messages before domain to response: {conversation.messages}"
            # )

            dtr = domain_to_response(
                conversation,
                ConvoFullResponse,
            )

            # logger.info(f"Convo messages after domain to response: {dtr.messages}")

            return dtr
        except Exception as e:
            raise e

    @staticmethod
    async def list_workspace_conversations(
        session: AsyncSession,
        user_id: str,
        workspace_id: str,
    ) -> list[ConvoListItemResponse]:
        logger.debug(f"Listing conversations for workspace {workspace_id}")

        try:
            conversations: list[ConvoFull] = await ConvoRepo().get_all_by_workspace_id(
                session=session,
                user_id=user_id,
                workspace_id=workspace_id,
            )

            logger.success(
                f"Retrieved {len(conversations)} conversations "
                f"for workspace {workspace_id}"
            )
            return [domain_to_response(c, ConvoListItemResponse) for c in conversations]

        except Exception as e:
            raise e

    @staticmethod
    async def update_conversation(
        session: AsyncSession,
        convo_update: ConvoUpdateRequest,
    ) -> ConvoFullResponse:
        logger.info(f"Updating conversation: {convo_update.convo_id}")

        try:
            conversation: ConvoFull = await ConvoRepo.update(
                session=session,
                convo_update=convo_update,
            )

            logger.success(f"Updated conversation: {convo_update.convo_id}")

            return domain_to_response(conversation, ConvoFullResponse)

        except Exception as e:
            raise e

    @staticmethod
    async def delete_conversation(
        session: AsyncSession,
        convo_request: ConvoDeleteRequest,
    ) -> bool:
        logger.info(f"Deleting conversation: {convo_request.convo_id}")

        deleted = await ConvoRepo.delete(
            session=session,
            convo_request=convo_request,
        )

        if deleted:
            logger.success(f"Deleted conversation: {convo_request.convo_id}")
        else:
            logger.warning(f"Conversation not found: {convo_request.convo_id}")

        return deleted

        # =========================================================================

    # Message Operations & RAG Logic
    # =========================================================================
    #
    #
    @staticmethod
    async def unpack_message(
        message: UIMessage, convo_id: str, user_id: str
    ) -> Message:
        if message.role == "user":
            _role = MessageRole.USER

        elif message.role == "assistant":
            _role = MessageRole.ASSISTANT
        else:
            _role = MessageRole.USER

        unpacked: Message = Message(
            role=_role,
            convo_id=convo_id,
            text="",
            message_id="placeholder",
            user_id=user_id,
        )

        for part in message.parts:
            if isinstance(part, TextPart) and part.type == "text":
                if not unpacked.text or unpacked.text == "":
                    unpacked.text += part.text
                else:
                    unpacked.text += "\n"
                    unpacked.text += part.text
            if isinstance(part, FilePart) and part.type == "file":
                file_part: FilePartDomain = pydantic_to_domain(part, FilePartDomain)

                if unpacked.attached_files:
                    unpacked.attached_files.append(file_part)
                else:
                    unpacked.attached_files = []
                    unpacked.attached_files.append(file_part)

        return unpacked

    @staticmethod
    async def save_message(
        session: AsyncSession,
        convo_id: str,
        user_id: str,
        message: UIMessage | MessageResponse,
    ) -> MessageResponse:
        logger.info(f"processing user message with RAG for conversation ")

        if isinstance(message, UIMessage):
            # unpack ui message to domain message
            msg: Message = await ConversationService.unpack_message(
                message, convo_id=convo_id, user_id=user_id
            )
            # save message
            save: MessageORM = await ConvoRepo.add_message(
                session=session,
                new_message=msg,
            )

            # return response to ui
            return domain_to_response(
                msg,
                MessageResponse,
            )

        message_in: Message = pydantic_to_domain(message, Message)

        db_msg: MessageORM = await ConvoRepo.add_message(
            session=session,
            new_message=message_in,
        )

        out_msg: Message = orm_to_domain(db_msg, Message)
        # NOTE: Sources added to metadata, but model should be updated with the sources field

        logger.success(f"Saved user message with RAG context")

        return domain_to_response(
            out_msg,
            MessageResponse,
        )
