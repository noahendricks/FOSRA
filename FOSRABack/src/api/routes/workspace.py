import sys
from pydantic import BaseModel
import msgspec
import json
from datetime import datetime
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
from starlette.types import Send
from FOSRABack.src.api.request_context import RequestContext
from FOSRABack.src.api.schemas.source_api_schemas import SourceGroupResponse
from FOSRABack.src.domain.enums import (
    ChunkerType,
    DocumentType,
    MessageRole,
    ParserType,
)
from FOSRABack.src.domain.schemas import (
    ChunkerConfig,
    EmbedderConfig,
    ParserConfig,
    RetrievedResult,
    SourceFull,
    StorageConfig,
    VectorStoreConfig,
)
from FOSRABack.src.services.conversation.llm_service import generate_llm_response
from FOSRABack.src.services.retrieval.vector_service import VectorService
from FOSRABack.src.services.workspace.workspace_service import WorkspaceService
from FOSRABack.src.api.schemas import (
    ConvoFullResponse,
    ConvoRequest,
    ConvoUpdateRequest,
    MessageResponse,
    NewConvoRequest,
    MessageRequest,
    SourceResponseDeep,
    WorkspaceFullResponse,
    WorkspaceRequest,
)
from FOSRABack.src.api.dependencies import get_db_session, get_session_factory
from typing import TYPE_CHECKING, Annotated, Any
from fastapi import APIRouter, Body, Depends, Query, Request

from FOSRABack.src.api.schemas.api_schemas import (
    ConvoDeleteRequest,
    ConvoListItemResponse,
    FilePart,
    NewConvoResponse,
    NewWorkspaceRequest,
    NewWorkspaceResponse,
    MessageRequest,
    TextPart,
    UIMessage,
    UIMessagePart,
    WorkspaceDeleteRequest,
    WorkspaceUpdateRequest,
)
from FOSRABack.src.services.conversation.conversation_service import ConversationService
from FOSRABack.src.storage.utils.converters import ulid_factory, domain_to_response
from FOSRABack.src.tasks.ingesting import ingest_files
from FOSRABack.src.tasks.processing import (
    chunk_sources,
    embed_documents,
    embed_query,
    parse_files,
)

import json

if TYPE_CHECKING:
    from langchain_core.messages import AIMessageChunk
    from litellm import AsyncIterator

from FOSRABack.src.services.retrieval.impls._utils import group_by_source, SourceGroup


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


class FileUpload(BaseModel):
    """Base message properties."""

    files: list[bytes]


@router.post("/file_upload")
async def intercept_file_binary(
    req: Annotated[FileUpload, Body()],
):
    # breakpoint()
    print(req)
    return None


# ============================================================================
# WORKSPACE
# ============================================================================

# ============================================================================
# GET
# ============================================================================


@router.get("/{user_id}/list_workspaces")
async def list_user_workspaces(
    user_id: str,
    session=Depends(get_db_session),
) -> list[WorkspaceFullResponse]:
    try:
        all_workspaces = await WorkspaceService().get_all_workspaces(
            user_id=user_id, session=session
        )

        logger.info(all_workspaces)

        return all_workspaces
    except Exception:
        raise


@router.get("/{user_id}/{workspace_id}/")
async def get_existing_workspace(
    request: Annotated[WorkspaceRequest, Query()], session=Depends(get_db_session)
) -> WorkspaceFullResponse:
    requested_workspace: WorkspaceFullResponse = (
        await WorkspaceService().retrieve_workspace_by_id(
            workspace_request=request, session=session
        )
    )
    return requested_workspace


# ============================================================================
# POST
# ============================================================================
@router.post("/{user_id}/create_workspace/")
async def new_workspace(
    request: Annotated[NewWorkspaceRequest, Query()], session=Depends(get_db_session)
) -> NewWorkspaceResponse:
    new_workspace: NewWorkspaceResponse = await WorkspaceService().create_workspace(
        create_workspace=request, session=session
    )
    return new_workspace


# ============================================================================
# PUT
# ============================================================================
@router.put("/{user_id}/{workspace_id}/")
async def update_workspace(
    request: Annotated[WorkspaceUpdateRequest, Query()],
    session=Depends(get_db_session),
) -> WorkspaceFullResponse:
    requested_workspace = await WorkspaceService().update_workspace(
        workspace_update=request, session=session
    )
    return requested_workspace


# ============================================================================
# DELETE
# ============================================================================
@router.delete("{user_id}/delete_workspaces/")
async def delete_workspaces(
    request: Annotated[WorkspaceDeleteRequest, Query()], session=Depends(get_db_session)
) -> bool:
    is_deleted = await WorkspaceService().delete_list_of_workspaces(
        workspace_request=request, session=session
    )

    return is_deleted


# ============================================================================
# Conversation
# ============================================================================


# ============================================================================
# GET
# ============================================================================
@router.get("/{user_id}/{convo_id}/get_convo")
async def get_convo(
    user_id: str, convo_id: str, session=Depends(get_db_session)
) -> ConvoFullResponse:
    try:
        convo: ConvoFullResponse = await ConversationService().get_conversation_by_id(
            user_id=user_id,
            convo_id=convo_id,
            session=session,
        )

        return convo
    except Exception as e:
        raise e


@router.get("/{user_id}/{workspace_id}/get_convos_list/")
async def get_list_of_convos(
    user_id: str, workspace_id: str, session=Depends(get_db_session)
) -> list[ConvoListItemResponse]:
    # breakpoint()
    convo_list: list[
        ConvoListItemResponse
    ] = await ConversationService().list_workspace_conversations(
        user_id=user_id,
        workspace_id=workspace_id,
        session=session,
    )

    return convo_list


# # GET Conversation metadata/settings
# @router.get("/{convo_id}/settings")
# async def get_conversation_settings(convo_id: str):
#     pass
#
# # GET Conversation config overrides
# @router.get("/conversation/{convo_id}")
# async def get_conversation_config(convo_id: str):
#     pass


# ============================================================================
# POST
# ============================================================================
@router.post("/user/profile")
async def new_temporary_convo(request):
    pass


@router.post("/{user_id}/{workspace_id}/new_convo/")
async def create_new_convo(
    request: Annotated[NewConvoRequest, Query()], session=Depends(get_db_session)
) -> NewConvoResponse:
    try:
        # TODO: Error Occurring but still storing; Potentially has to do with workspace id and values passed
        new_convo: NewConvoResponse = await ConversationService().create_conversation(
            new_convo=request, session=session
        )
        return new_convo

    except Exception as e:
        raise e


# ============================================================================
# PUT
# ============================================================================
@router.put("/{workspace_id}/{convo_id}")
async def update_convo(
    request: Annotated[ConvoUpdateRequest, Query()], session=Depends(get_db_session)
) -> ConvoFullResponse:
    try:
        # WARN: Not Correct; Need to implement
        convo_update = await ConversationService().update_conversation(
            session=session, convo_update=request
        )

        return convo_update
    except Exception as e:
        raise e


@router.post("/{convo_id}/archive/")
async def archive_conversation(
    request: Annotated[ConvoRequest, Query()], session=Depends(get_db_session)
) -> list[str]:
    try:
        archived_convos: list[str] = await WorkspaceService().archive_convo(
            convo_request=request, session=session
        )
        return archived_convos
    except Exception as e:
        raise e


@router.post("/{convo_id}/restore/")
async def restore_conversation(
    request: Annotated[ConvoRequest, Query()], session=Depends(get_db_session)
) -> list[str]:
    try:
        restored_convos: list[str] = await WorkspaceService().restore_convo(
            convo_request=request, session=session
        )
        # TODO: Add is_archived to ORM Models and Schemas
        return restored_convos
    except Exception as e:
        raise e


# # PUT Update conversation settings (your toggles)
# @router.put("/{convo_id}/settings")
# async def update_conversation_settings(convo_id: str, request):
#     pass


# # PUT Update conversation config overrides
# @router.put("/conversation/{convo_id}")
# async def update_conversation_config(convo_id: str, request):
#     pass


# ============================================================================
# DELETE
# ============================================================================
# - DELETE Convo
@router.delete("/user/profile/{user_id}")
async def delete_temporary_convo(request):
    pass


@router.delete("/{workspace_id}/{convo_id}")
async def delete_convo(
    request: Annotated[ConvoDeleteRequest, Query()], session=Depends(get_db_session)
) -> bool:
    try:
        convo_update: bool = await ConversationService().delete_conversation(
            session=session, convo_request=request
        )

        return convo_update
    except Exception as e:
        raise e


def extract_text_from_parts(parts: list[UIMessagePart]) -> str:
    text_parts = []
    file_parts: dict[str, dict[str, Any]] = {}
    for part in parts:
        if isinstance(part, TextPart) and part.type == "text":
            text_parts.append(part.text)
        if isinstance(part, FilePart) and part.type == "file":
            file_parts[part.filename if part.filename else ""] = {
                "url": part.url,
                "mediaType": part.media_type,
            }

    return "\n".join(text_parts)


@router.post("/{convo_id}/send_message/")
async def send_message_stream(
    req: MessageRequest,
    db_session=Depends(get_db_session),
    session_factory=Depends(get_session_factory),
):
    async def stream():
        try:
            async with session_factory() as session:
                logger.debug("entry")
                if not req.convo_id:
                    raise ValueError("No ConvoId Provided in Attempt to Send Message")

                ctx = await RequestContext.from_request(
                    user_id=req.user_id,
                    workspace_id=req.workspace_id,
                    convo_id=req.convo_id,
                    session=session,
                )

                text_part_id: str = ulid_factory()

                cleaned_messages: list[UIMessage] = []

                new_message = req.messages[-1]

                user_query: str = extract_text_from_parts(req.messages[-1].parts)

                def emit_chunk(chunk: dict[str, Any]) -> str:
                    return f"data: {json.dumps(chunk)}\n\n"

                storage_config = StorageConfig()
                parser_config = ParserConfig(preferrend_parser_type=ParserType.MARKDOWN)
                chunker_config = ChunkerConfig(preferred_chunker_type=ChunkerType.TOKEN)
                embedder_config = EmbedderConfig()
                vector_config = VectorStoreConfig(
                    host="localhost", port=6333, api_base=None
                )

                # TODO: Add retrieving files from user message source part

                logger.debug("prior to message save")

                message: MessageResponse = await ConversationService().save_message(
                    message=new_message,
                    convo_id=ctx.convo_id if ctx.convo_id else "",
                    session=db_session,
                    user_id=ctx.user_id,
                )

                logger.debug("after message save")

                message_id: str = message.message_id if message.message_id else ""

                logger.debug("prior to ingest")

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "fetching", "progress": 0.1},
                    }
                )

                logger.debug(sys.getsizeof(new_message))

                print()

                files = await ingest_files(
                    files=[i for i in new_message.parts if isinstance(i, FilePart)],
                    session=session,
                    storage_config=storage_config,
                )

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "processing", "progress": 0.25},
                    }
                )

                print(
                    "DEBUGPRINT[70]: workspace.py:457 (before parsed = await parse_files()"
                )
                parsed = await parse_files(
                    files_list=files,
                    config=parser_config,
                    session_factory=session_factory,
                )
                print(f"DEBUGPRINT[71]: workspace.py:458: parsed={parsed}")

                chunked = await chunk_sources(config=chunker_config, sources=parsed)

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "chunking", "progress": 0.35},
                    }
                )

                embedded: list[SourceFull] = await embed_documents(
                    sources=chunked,
                    config=embedder_config,
                    session_factory=session_factory,
                )

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "embedding", "progress": 0.45},
                    }
                )

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "storing vectors", "progress": 0.55},
                    }
                )

                upsert = await VectorService().upsert(
                    sources=embedded,
                    config=vector_config,
                    session_factory=session_factory,
                )

                vector_config.config_name = "test config"
                vector_config.config_id = 1

                logger.debug("after ingest, before search")

                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "searching", "progress": 0.75},
                    }
                )

                query = await embed_query(
                    query=user_query,
                    config=embedder_config,
                    session=session,
                )

                search: list[RetrievedResult] = await VectorService().search(
                    query_vector=query,
                    query_text=user_query,
                    config=vector_config,
                    session=session,
                )

                logger.debug("after search")

                # NOTE: Currently throwing: "expected str got array"
                sources_grouped: list[SourceGroup] = group_by_source(results=search)
                print()

                source_groups_response: list[SourceGroupResponse] = [
                    domain_to_response(group, SourceGroupResponse)
                    for group in sources_grouped
                ]
                # NOTE: EMPTY

                sources_as_dicts: list[dict[str, Any]] = [
                    group.model_dump(mode="json") for group in source_groups_response
                ]

                print()
                # NOTE: EMPTY
                yield emit_chunk(
                    {
                        "type": "data-rag-status",
                        "data": {"stage": "complete", "progress": 1},
                    }
                )

                logger.debug(f"SOURCES AFTER ENCODED TO STRING")

                sources_json = json.dumps(
                    [group.model_dump(mode="json") for group in source_groups_response]
                )
                # NOTE: EMPTY

                print()
                # NOTE: EMPTY

                for src in sources_as_dicts:
                    yield emit_chunk(
                        {
                            "type": "rag-source",
                            "source": src,
                        }
                    )
                yield emit_chunk({"type": "start", "messageId": message_id})
                yield emit_chunk({"type": "text-start", "id": text_part_id})

            # TODO: Add LLM Initialization checks and remediation
            # NOTE: Type narrowing should have got here, messages shouldn't be none

            stream: AsyncIterator[AIMessageChunk] = await generate_llm_response(
                chat_history=req.messages if req.messages else [],
                convo_id=req.convo_id,
                sources=sources_grouped,
                user_prefs=None,
            )

            full_text = ""

            logger.debug("prior to stream")
            async for chunk in stream:
                content = chunk.content
                if content:
                    text_chunk: str = (
                        content if isinstance(content, str) else str(content)
                    )
                    full_text += text_chunk

                    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_part_id, 'delta': text_chunk})}\n\n"

            logger.debug("after stream")

            yield f"data: {json.dumps({'type': 'text-end', 'id': text_part_id})}\n\n"

            logger.info(sources_grouped)

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
            # TODO: yield assistant message id

            yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
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


@router.post("/stub/stub")
async def zod_stub_sources(
    convo_id: str,
    request: Request,
    db_session=Depends(get_db_session),
    session_factory=Depends(get_session_factory),
) -> SourceGroupResponse:
    retrieval_result = SourceGroupResponse(
        chunk_count=0,
        chunks=[],
        source=SourceResponseDeep(
            document_type=DocumentType.DOC,
            hash="",
            metadata={},
            name="",
            origin_path="",
            origin_type="",
            result_score=0,
            source_id="",
            source_summary="",
            summary_embedding="",
        ),
        top_score=0,
    )

    return retrieval_result
