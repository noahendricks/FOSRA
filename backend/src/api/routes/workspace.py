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
    WorkspaceFullResponse,
    WorkspaceRequest,
)
from backend.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoListItemResponse,
    NewConvoResponse,
    NewWorkspaceRequest,
    NewWorkspaceResponse,
    TextPart,
    UIMessagePart,
    WorkspaceDeleteRequest,
    WorkspaceUpdateRequest,
)
from backend.src.api.schemas.source_api_schemas import (
    ChunkResponse,
    ChunkWithScoreResponse,
    SourceGroupResponse,
)
from backend.src.domain.enums import DocumentType, MessageRole
from backend.src.domain.schemas.config import ScoredRetrieval
from backend.src.services.conversation.agent_service import create_fosra_agent
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.conversation.utils.llm_utils import ui_messages_to_lc_messages
from backend.src.services.retrieval.vector_service import RetrievedChunk
from backend.src.services.workspace.workspace_service import WorkspaceService
from backend.src.storage.utils.converters import ulid_factory


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


# ======================================================================
# Helpers
# ======================================================================


class FileUpload(BaseModel):
    """Base message properties."""

    files: list[bytes]


def extract_text_from_parts(parts: list[UIMessagePart]) -> str:
    """Extract plain text from a list of UI message parts."""
    text_parts = []
    for part in parts:
        if isinstance(part, TextPart) and part.type == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts)


def _retrieved_chunk_to_scored(chunk: RetrievedChunk, rank: int) -> ScoredRetrieval:
    """Convert a ``RetrievedChunk`` (from vector search / reranking) into
    the ``ScoredRetrieval`` shape that LLMService and citation formatting
    expect.
    """
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
    """Build ``SourceGroupResponse`` objects grouped by source_id."""
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
                    document_type=DocumentType.DOC,
                    result_score=top_score,
                ),
                chunks=chunk_responses,
                top_score=top_score,
                chunk_count=len(chunk_responses),
            )
        )

    return result


# ======================================================================
# File upload (placeholder)
# ======================================================================


@router.post("/file_upload")
async def intercept_file_binary(
    req: Annotated[FileUpload, Body()],
):
    logger.debug("file_upload received {} files", len(req.files))
    return None


# ======================================================================
# Workspace CRUD
# ======================================================================


@router.get("/{user_id}/list_workspaces")
async def list_user_workspaces(
    user_id: str,
    session=Depends(get_db_session),
) -> list[WorkspaceFullResponse]:
    all_workspaces = await WorkspaceService().get_all_workspaces(
        user_id=user_id, session=session
    )
    return all_workspaces


@router.get("/{user_id}/{workspace_id}/")
async def get_existing_workspace(
    request: Annotated[WorkspaceRequest, Query()], session=Depends(get_db_session)
) -> WorkspaceFullResponse:
    return await WorkspaceService().retrieve_workspace_by_id(
        workspace_request=request, session=session
    )


@router.post("/{user_id}/create_workspace/")
async def new_workspace(
    request: Annotated[NewWorkspaceRequest, Query()], session=Depends(get_db_session)
) -> NewWorkspaceResponse:
    return await WorkspaceService().create_workspace(
        create_workspace=request, session=session
    )


@router.put("/{user_id}/{workspace_id}/")
async def update_workspace(
    request: Annotated[WorkspaceUpdateRequest, Query()],
    session=Depends(get_db_session),
) -> WorkspaceFullResponse:
    return await WorkspaceService().update_workspace(
        workspace_update=request, session=session
    )


@router.delete("{user_id}/delete_workspaces/")
async def delete_workspaces(
    request: Annotated[WorkspaceDeleteRequest, Query()],
    session=Depends(get_db_session),
) -> bool:
    return await WorkspaceService().delete_list_of_workspaces(
        workspace_request=request, session=session
    )


# ======================================================================
# Conversation CRUD
# ======================================================================


@router.get("/{user_id}/{convo_id}/get_convo")
async def get_convo(
    user_id: str, convo_id: str, session=Depends(get_db_session)
) -> ConvoFullResponse:
    return await ConversationService().get_conversation_by_id(
        user_id=user_id, convo_id=convo_id, session=session
    )


@router.get("/{user_id}/{workspace_id}/get_convos_list/")
async def get_list_of_convos(
    user_id: str, workspace_id: str, session=Depends(get_db_session)
) -> list[ConvoListItemResponse]:
    return await ConversationService().list_workspace_conversations(
        user_id=user_id, workspace_id=workspace_id, session=session
    )


@router.post("/user/profile")
async def new_temporary_convo(request):
    pass


@router.post("/{user_id}/{workspace_id}/new_convo/")
async def create_new_convo(
    request: Annotated[NewConvoRequest, Query()], session=Depends(get_db_session)
) -> NewConvoResponse:
    return await ConversationService().create_conversation(
        new_convo=request, session=session
    )


@router.put("/{workspace_id}/{convo_id}")
async def update_convo(
    request: Annotated[ConvoUpdateRequest, Query()], session=Depends(get_db_session)
) -> ConvoFullResponse:
    # WARN: Not fully implemented yet
    return await ConversationService().update_conversation(
        session=session, convo_update=request
    )


@router.post("/{convo_id}/archive/")
async def archive_conversation(
    request: Annotated[ConvoRequest, Query()], session=Depends(get_db_session)
) -> list[str]:
    return await WorkspaceService().archive_convo(
        convo_request=request, session=session
    )


@router.post("/{convo_id}/restore/")
async def restore_conversation(
    request: Annotated[ConvoRequest, Query()], session=Depends(get_db_session)
) -> list[str]:
    # TODO: Add is_archived to ORM Models and Schemas
    return await WorkspaceService().restore_convo(
        convo_request=request, session=session
    )


@router.delete("/user/profile/{user_id}")
async def delete_temporary_convo(request):
    pass


@router.delete("/{workspace_id}/{convo_id}")
async def delete_convo(
    request: Annotated[ConvoDeleteRequest, Query()], session=Depends(get_db_session)
) -> bool:
    return await ConversationService().delete_conversation(
        session=session, convo_request=request
    )


# ======================================================================
# Chat  —  send_message_stream
# ======================================================================


@router.post("/{convo_id}/send_message/")
async def send_message_stream(
    req: MessageRequest,
    db_session=Depends(get_db_session),
    session_factory=Depends(get_session_factory),
):
    """SSE endpoint: save user message, run FOSRA agent, stream response.

    The agent (DeepAgents + LangGraph) handles retrieval internally via the
    ``search_knowledge_base`` tool.  The route streams token-level LLM output
    as ``text-delta`` SSE events, then emits ``rag-source`` events from the
    retrieval result store after the agent finishes.
    """

    async def stream():
        try:
            async with session_factory() as session:
                if not req.convo_id:
                    raise ValueError("No convo_id provided")

                # -- Context bundle ----------------------------------------
                ctx = await RequestContext.from_request(
                    user_id=req.user_id,
                    workspace_id=req.workspace_id,
                    convo_id=req.convo_id,
                    session=session,
                )

                text_part_id: str = ulid_factory()
                user_prefs = ctx.preferences

                # -- 1. Save user message ----------------------------------
                new_message = req.messages[-1]
                message: MessageResponse = await ConversationService().save_message(
                    message=new_message,
                    convo_id=ctx.convo_id or "",
                    session=db_session,
                    user_id=ctx.user_id,
                )
                message_id: str = message.message_id or ""

                # -- 2. Create agent ---------------------------------------
                agent, result_store = create_fosra_agent(user_prefs)

                # -- 3. Build LangChain messages from UI messages ----------
                lc_messages = ui_messages_to_lc_messages(req.messages or [])

                # -- 4. Stream agent output --------------------------------
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

                # -- 5. Emit source groups from retrieval result store ------
                source_groups = _chunks_to_source_groups(result_store.chunks)
                sources_as_dicts: list[dict[str, Any]] = [
                    group.model_dump(mode="json") for group in source_groups
                ]

                for src in sources_as_dicts:
                    yield emit_chunk({"type": "rag-source", "source": src})

                # -- 6. Save assistant message ------------------------------
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


"""
AI SDK UIMessageChunk types (for reference):

1. Message lifecycle:
   - type: 'start' - Message begins (optional messageId, messageMetadata)
   - type: 'finish' - Message ends (optional finishReason, messageMetadata)
   - type: 'abort' - Stream was aborted

2. Text streaming:
   - type: 'text-start' - Text part begins (requires id)
   - type: 'text-delta' - Text chunk (requires id, delta)
   - type: 'text-end' - Text part ends (requires id)

3. Reasoning (optional):
   - type: 'reasoning-start' - Reasoning begins
   - type: 'reasoning-delta' - Reasoning chunk
   - type: 'reasoning-end' - Reasoning ends

4. Sources:
   - type: 'source-url' - URL source (requires sourceId, url)
   - type: 'source-document' - Document source (requires sourceId, mediaType, title)

5. Files:
   - type: 'file' - File attachment (requires url, mediaType)

6. Tools (if using):
   - type: 'tool-input-start' - Tool call begins
   - type: 'tool-input-delta' - Tool input streaming
   - type: 'tool-input-available' - Tool input complete
   - type: 'tool-output-available' - Tool output ready

7. Custom data:
   - type: 'data-{name}' - Custom data part

8. Errors:
   - type: 'error' - Error occurred (requires errorText)

9. Steps (for multi-step):
   - type: 'start-step' - Step begins
   - type: 'finish-step' - Step ends
"""
