from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client.conversions.common_types import QueryResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas import (
    ConvoFullResponse,
    ConvoUpdateRequest,
    MessageResponse,
    NewConvoRequest,
)
from backend.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoListItemResponse,
    FilePart,
    NewConvoResponse,
    TextPart,
    UIMessage,
)
from backend.src.domain.enums import MessageRole, VectorStoreType
from backend.src.domain.schemas.config import ScoredRetrieval
from backend.src.domain.schemas.convo import Convo, Message, NewConvo
from backend.src.domain.schemas.doc import MDNFile
from backend.src.storage.convo import ConvoRepo
from backend.src.storage.utils.converters import (
    domain_to_response,
    orm_to_domain,
    pydantic_to_domain,
)

if TYPE_CHECKING:
    from backend.src.storage.models import MessageORM


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
            conversation: NewConvo = await ConvoRepo().create(
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

            conversation: Convo = await ConvoRepo.get_by_id(
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
            conversations: list[Convo] = await ConvoRepo().get_all_by_workspace_id(
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
            conversation: Convo = await ConvoRepo.update(
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
            parent_id=message.message_metadata.get("parent_id")
            if message.message_metadata
            else None,
            root_id=message.message_metadata.get("root_id")
            if message.message_metadata
            else None,
        )

        for part in message.parts:
            if isinstance(part, TextPart) and part.type == "text":
                if not unpacked.text or unpacked.text == "":
                    unpacked.text += part.text
                else:
                    unpacked.text += "\n"
                    unpacked.text += part.text
            if isinstance(part, MDNFile) and part.type == "file":
                # FIX:
                if unpacked.attached_files:
                    unpacked.attached_files.append(part)
                else:
                    unpacked.attached_files = []
                    unpacked.attached_files.append(part)

        return unpacked

    @staticmethod
    async def save_message(
        session: AsyncSession,
        convo_id: str,
        user_id: str,
        message: UIMessage | MessageResponse,
    ) -> MessageResponse:
        logger.info("processing user message with RAG for conversation ")

        if isinstance(message, UIMessage):
            # unpack ui message to domain message
            msg: Message = await ConversationService.unpack_message(
                message, convo_id=convo_id, user_id=user_id
            )
            # save message
            _: MessageORM = await ConvoRepo.add_message(
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

        logger.success("Saved user message with RAG context")

        return domain_to_response(
            out_msg,
            MessageResponse,
        )

    @staticmethod
    async def parse_retrievals(retrievals: QueryResponse, store_type: VectorStoreType):
        sources = []

        match store_type:
            case VectorStoreType.QDRANT:
                points = retrievals.points

                for p in points:
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


# {'id': 'db121816-9a37-4e4b-95e6-cfeb5c9717a4',
#              'version': 1,
#              'score': 7.6365137,
#              'payload': {'chunk': 'But his\r\n'
#                                   'disobedient hands gave no heed to the '
#                                   'command. They beat the water\r\n'
#                                   'vigorously with quick, downward strokes, '
#                                   'forcing him to the surface. He\r\n'
#                                   'felt his head emerge; his eyes were blinded '
#                                   'by the sunlight; his chest\r\n'
#                                   'expanded convulsively, and with a supreme '
#                                   'and crowning agony his lungs\r\n'
#                                   'engulfed a great draught of air, which '
#                                   'instantly he expelled in a\r\n'
#                                   'shriek!\r\n'
#                                   '\r\n'
#                                   'He was now in full possession of his '
#                                   'physical senses. They were,\r\n'
#                                   'indeed, preternaturally keen and alert. '
#                                   'Something in the awful\r\n'
#                                   'disturbance of his organic system had so '
#                                   'exalted and refined them that\r\n'
#                                   'they made record of things never before '
#                                   'perceived. He felt the ripples\r\n'
#                                   'upon his face and heard their separate '
#                                   'sounds as they struck. He looked\r\n'
#                                   'at the forest on the bank of the stream, '
#                                   'saw the individual trees, the\r\n'
#                                   'leaves and the veining of each leaf—he saw '
#                                   'the very insects upon them:\r\n'
#                                   'the locusts, the brilliant bodied flies, '
#                                   'the gray spiders stretching\r\n'
#                                   'their webs from twig to twig. He noted the '
#                                   'prismatic colors in all the\r\n'
#                                   'dewdrops upon a million blades of grass. '
#                                   'The humming of the gnats that\r\n'
#                                   'danced above the eddies of the stream, the '
#                                   'beating of the dragon flies’\r\n'
#                                   'wings, the strokes of the water spiders’ '
#                                   'legs, like oars which had\r\n'
#                                   'lifted their boat—all these made audible '
#                                   'music. A fish slid along\r\n'
#                                   'beneath his eyes and he heard the rush of '
#                                   'its body parting the water.\r\n'
#                                   '\r\n'
#                                   'He had come to the surface facing down the '
#                                   'stream; in a moment the\r\n'
#                                   'visible world seemed to wheel slowly round, '
#                                   'himself the pivotal point,\r\n'
#                                   'and he saw the bridge, the fort, the '
#                                   'soldiers upon the bridge, the\r\n'
#                                   'captain, the sergeant, the two privates, '
#                                   'his executioners. They were in\r\n'
#                                   'silhouette against the blue sky. They '
#                                   'shouted and gesticulated,\r\n'
#                                   'pointing at him. The captain had drawn his '
#                                   'pistol, but did not fire;\r\n'
#                                   'the others were unarmed. Their movements '
#                                   'were grotesque and horrible,\r\n'
#                                   'their forms gigantic.\r\n'
#                                   '\r\n'
#                                   'Suddenly he heard a sharp report and '
#                                   'something struck the water smartly\r\n'
#                                   'within a few inches of his head, spattering '
#                                   'his face with spray. He\r\n'
#                                   'heard a second report, and saw one of the '
#                                   'sentinels with his rifle at\r\n'
#                                   'his shoulder, a light cloud of blue smoke '
#                                   'rising from the muzzle. The\r\n'
#                                   'man in the water saw the eye of the man on '
#                                   'the bridge gazing into his\r\n'
#                                   'own through the sights of the rifle. He '
#                                   'observed that it was a gray eye\r\n'
#                                   'and remembered having read that gray eyes '
#                                   'were keenest, and that all\r\n'
#                                   'famous marksmen had them. '},
#              'vector': None,
#              'shard_key': None,
#              'order_value': None}]}
