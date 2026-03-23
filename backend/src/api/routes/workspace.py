from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from loguru import logger
from pydantic import BaseModel

from backend.src.api.dependencies import get_db_session, get_session_factory
from backend.src.api.request_context import RequestContext
from backend.src.api.schemas import (
    ConvoFullResponse,
    ConvoRequest,
    ConvoUpdateRequest,
    MessageRequest,
    MessageResponse,
    NewConvoRequest,
    SourceResponseDeep,
)
from backend.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoListItemResponse,
    NewConvoResponse,
    TextPart,
    UIMessagePart,
)
from backend.src.api.schemas.source_api_schemas import (
    ChunkResponse,
    ChunkWithScoreResponse,
    SourceGroupResponse,
)
from backend.src.domain.enums import DocType, MessageRole
from backend.src.settings import ScoredRetrieval
from backend.src.services.conversation.agent_service import create_fosra_agent
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.conversation.utils.llm_utils import ui_messages_to_lc_messages
from backend.src.services.retrieval.vector_service import RetrievedChunk
from backend.src.storage.utils.converters import ulid_factory


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


class FileUpload(BaseModel):
    files: list[bytes]


def extract_text_from_parts(parts: list[UIMessagePart]) -> str:
    text_parts = []
    for part in parts:
        if isinstance(part, TextPart) and part.type == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts)


def _retrieved_chunk_to_scored(chunk: RetrievedChunk, rank: int) -> ScoredRetrieval:
    return ScoredRetrieval(
        rank=rank,
        score=chunk.score,
        text=chunk.text,
        doc_title=chunk.payload.get("doc_title", ""),
        chunk_id=chunk.payload.get("chunk_id", str(rank)),
        doc_id=chunk.payload.get("source_id", ""),
        page_number=chunk.payload.get("page_number", 0),
        start_index=chunk.start_char,
        end_index=chunk.payload.get("end_char", chunk.start_char + len(chunk.text)),
    )


def _chunks_to_source_groups(
    chunks: list[RetrievedChunk],
) -> list[SourceGroupResponse]:
    from collections import defaultdict

    groups: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        source_id = chunk.payload.get("source_id", "unknown")
        groups[source_id].append((idx, chunk))

    result: list[SourceGroupResponse] = []
    for source_id, items in groups.items():
        chunk_responses: list[ChunkWithScoreResponse] = []
        top_score = 0.0
        for idx, chunk in items:
            top_score = max(top_score, chunk.score)
            chunk_responses.append(
                ChunkWithScoreResponse(
                    chunk=ChunkResponse(
                        chunk_id=chunk.payload.get("chunk_id", str(idx)),
                        source_id=source_id,
                        source_hash="",
                        start_index=chunk.start_char,
                        end_index=chunk.payload.get(
                            "end_char", chunk.start_char + len(chunk.text)
                        ),
                        token_count=chunk.token_count,
                        text=chunk.text,
                    ),
                    similarity_score=chunk.score,
                    reranker_score=chunk.score,
                )
            )

        result.append(
            SourceGroupResponse(
                source=SourceResponseDeep(
                    id=source_id,
                    type=None,
                    name=source_id,
                    document_type=DocType.DOC,
                    result_score=top_score,
                ),
                chunks=chunk_responses,
                top_score=top_score,
                chunk_count=len(chunk_responses),
            )
        )

    return result


@router.post("/file_upload")
async def intercept_file_binary(
    req: Annotated[FileUpload, Body()],
):
    logger.debug("file_upload received {} files", len(req.files))
    return None


@router.get("/{user_id}/{convo_id}/get_convo")
async def get_convo(
    user_id: str, convo_id: str, session=Depends(get_db_session)
) -> ConvoFullResponse:
    return await ConversationService().get_conversation_by_id(
        user_id=user_id, convo_id=convo_id, session=session
    )


@router.get("/{user_id}/get_convos_list/")
async def get_list_of_convos(
    user_id: str, session=Depends(get_db_session)
) -> list[ConvoListItemResponse]:
    return await ConversationService().list_conversations(
        user_id=user_id, session=session
    )


@router.post("/{user_id}/new_convo/")
async def create_new_convo(
    request: Annotated[NewConvoRequest, Query()], session=Depends(get_db_session)
) -> NewConvoResponse:
    return await ConversationService().create_conversation(
        new_convo=request, session=session
    )


@router.put("/{convo_id}")
async def update_convo(
    request: Annotated[ConvoUpdateRequest, Query()], session=Depends(get_db_session)
) -> ConvoFullResponse:
    return await ConversationService().update_conversation(
        session=session, convo_update=request
    )


@router.delete("/{convo_id}")
async def delete_convo(
    request: Annotated[ConvoDeleteRequest, Query()], session=Depends(get_db_session)
) -> bool:
    return await ConversationService().delete_conversation(
        session=session, convo_request=request
    )


@router.post("/{convo_id}/send_message/")
async def send_message_stream(
    req: MessageRequest,
    db_session=Depends(get_db_session),
    session_factory=Depends(get_session_factory),
):
    async def stream():
        try:
            async with session_factory() as session:
                if not req.convo_id:
                    raise ValueError("No convo_id provided")

                ctx = await RequestContext.from_request(
                    user_id=req.user_id,
                    convo_id=req.convo_id,
                    session=session,
                )

                text_part_id: str = ulid_factory()
                user_prefs = ctx.preferences

                new_message = req.messages[-1]
                message: MessageResponse = await ConversationService().save_message(
                    message=new_message,
                    convo_id=ctx.convo_id or "",
                    session=db_session,
                    user_id=ctx.user_id,
                )
                message_id: str = message.message_id or ""

                agent, result_store = create_fosra_agent(user_prefs)

                lc_messages = ui_messages_to_lc_messages(req.messages or [])

                def emit_chunk(chunk: dict[str, Any]) -> str:
                    return f"data: {json.dumps(chunk)}\n\n"

                yield emit_chunk({"type": "start", "messageId": message_id})
                yield emit_chunk({"type": "text-start", "id": text_part_id})

                full_text = ""

                async for msg, _metadata in agent.astream(
                    {"messages": lc_messages},
                    stream_mode="messages",
                ):
                    if isinstance(msg, AIMessageChunk) and msg.content:
                        text_chunk = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        full_text += text_chunk
                        yield emit_chunk(
                            {
                                "type": "text-delta",
                                "id": text_part_id,
                                "delta": text_chunk,
                            }
                        )

                yield emit_chunk({"type": "text-end", "id": text_part_id})

                source_groups = _chunks_to_source_groups(result_store.items)
                sources_as_dicts: list[dict[str, Any]] = [
                    group.model_dump(mode="json") for group in source_groups
                ]

                for src in sources_as_dicts:
                    yield emit_chunk({"type": "rag-source", "source": src})

                _ = await ConversationService().save_message(
                    message=MessageResponse(
                        role=MessageRole.ASSISTANT,
                        text=full_text,
                        user_id=req.user_id,
                        attached_sources=sources_as_dicts,
                        convo_id=req.convo_id,
                    ),
                    convo_id=req.convo_id,
                    user_id=req.user_id,
                    session=session,
                )

                yield emit_chunk({"type": "finish", "finishReason": "stop"})

        except Exception as e:
            logger.error("Stream error: {}", e)
            yield f"data: {json.dumps({'type': 'error', 'errorText': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(path="/{convo_id}/reconnect/")
async def reconnect_to_stream(convo_id: str, request: Request):
    async def empty_stream():
        yield ""

    return StreamingResponse(
        empty_stream(),
        media_type="text/event-stream",
    )
