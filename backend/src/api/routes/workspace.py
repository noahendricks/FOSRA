from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.dependencies import get_db_session
from backend.src.api.schemas.session_api_schemas import (
    NewSessionRequest,
    NewSessionResponse,
    SessionDeleteRequest,
    SessionFullResponse,
    SessionListItemResponse,
    SessionUpdateRequest,
)
from backend.src.api.schemas.source_api_schemas import SourceResponseDeep
from backend.src.api.schemas.tui_control_schemas import (
    TextPart,
    UIMessagePart,
)
from backend.src.api.schemas.source_api_schemas import (
    ChunkResponse,
    ChunkWithScoreResponse,
    SourceGroupResponse,
)
from backend.src.domain.enums import DocType
from backend.src.settings import ScoredRetrieval
from backend.src.services.session.conversation_service import SessionService
from backend.src.services.retrieval.vector_service import RetrievedChunk


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
        score=chunk.score,  # type: ignore[reportUnknownMemberType]
        text=chunk.text,  # type: ignore[reportUnknownMemberType]
        doc_title=chunk.payload.get("doc_title", ""),  # type: ignore[reportUnknownMemberType]
        chunk_id=chunk.payload.get("chunk_id", str(rank)),  # type: ignore[reportUnknownMemberType]
        doc_id=chunk.payload.get("source_id", ""),  # type: ignore[reportUnknownMemberType]
        page_number=chunk.payload.get("page_number", 0),  # type: ignore[reportUnknownMemberType]
        start_index=chunk.start_char,  # type: ignore[reportUnknownMemberType]
        end_index=chunk.payload.get("end_char", chunk.start_char + len(chunk.text)),  # type: ignore[reportUnknownMemberType]
    )


def _chunks_to_source_groups(
    chunks: list[RetrievedChunk],
) -> list[SourceGroupResponse]:
    from collections import defaultdict

    groups: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        source_id = chunk.payload.get("source_id", "unknown")  # type: ignore[reportUnknownMemberType]
        groups[source_id].append((idx, chunk))

    result: list[SourceGroupResponse] = []
    for source_id, items in groups.items():
        chunk_responses: list[ChunkWithScoreResponse] = []
        top_score = 0.0
        for idx, chunk in items:
            top_score = max(top_score, chunk.score)  # type: ignore[reportUnknownMemberType]
            chunk_responses.append(
                ChunkWithScoreResponse(
                    chunk=ChunkResponse(
                        chunk_id=chunk.payload.get("chunk_id", str(idx)),  # type: ignore[reportUnknownMemberType]
                        source_id=source_id,
                        source_hash="",
                        start_index=chunk.start_char,  # type: ignore[reportUnknownMemberType]
                        end_index=chunk.payload.get(  # type: ignore[reportUnknownMemberType]
                            "end_char",
                            chunk.start_char + len(chunk.text),  # type: ignore[reportUnknownMemberType]
                        ),
                        token_count=chunk.token_count,  # type: ignore[reportUnknownMemberType]
                        text=chunk.text,  # type: ignore[reportUnknownMemberType]
                    ),
                    similarity_score=chunk.score,  # type: ignore[reportUnknownMemberType]
                    reranker_score=chunk.score,  # type: ignore[reportUnknownMemberType]
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


@router.get("/{user_id}/{session_id}/get_session")
async def get_session(
    user_id: str,
    session_id: str,
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
) -> SessionFullResponse:
    return await SessionService().get_session_by_id(
        user_id=user_id, session_id=session_id, session=session
    )


@router.get("/{user_id}/get_sessions_list/")
async def get_list_of_sessions(
    user_id: str,
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
) -> list[SessionListItemResponse]:
    return await SessionService().list_sessions(user_id=user_id, session=session)


@router.post("/{user_id}/new_session/")
async def create_new_session(
    request: Annotated[NewSessionRequest, Query()],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
) -> NewSessionResponse:
    return await SessionService().create_session(new_session=request, session=session)


@router.put("/{session_id}")
async def update_session(
    request: Annotated[SessionUpdateRequest, Query()],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
) -> SessionFullResponse:
    return await SessionService().update_session(
        session=session, session_update=request
    )


@router.delete("/{session_id}")
async def delete_session(
    request: Annotated[SessionDeleteRequest, Query()],
    session: AsyncSession = Depends(get_db_session),  # type: ignore[reportExplicitAny]
) -> bool:
    return await SessionService().delete_session(
        session=session, session_request=request
    )


# Deprecated endpoints removed — use /oc routes instead
