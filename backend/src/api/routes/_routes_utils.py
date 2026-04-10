"""Shared utilities for API routes."""

from __future__ import annotations

from collections import defaultdict

from backend.src.api.schemas.source_api_schemas import (
    ChunkResponse,
    ChunkWithScoreResponse,
    SourceGroupResponse,
    SourceResponseDeep,
)
from backend.src.services.retrieval.vector_service import RetrievedChunk


def _chunks_to_source_groups(
    chunks: list[RetrievedChunk],
) -> list[SourceGroupResponse]:
    """group retrieved chunks by source_id and return as SourceGroupResponse.

    this is used by workspace and retrieval routes to organize chunks
    grouped by their source document for ranked retrieval results.

    args:
        chunks: List of retrieved chunks from vector/graph search

    returns:
        List of SourceGroupResponse, one per unique source
    """
    groups: dict[str, list[tuple[int, RetrievedChunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        source_id = chunk.payload.get("source_id", "unknown")
        groups[source_id].append((idx, chunk))

    result: list[SourceGroupResponse] = []
    for source_id, items in groups.items():
        chunk_with_scores: list[ChunkWithScoreResponse] = []
        top_score = 0.0
        for idx, chunk in items:
            top_score = max(top_score, chunk.score)
            chunk_with_scores.append(
                ChunkWithScoreResponse(
                    chunk=ChunkResponse(
                        chunk_id=chunk.payload.get("chunk_id", str(idx)),
                        source_id=source_id,
                        source_hash="",
                        start_index=chunk.start_char,
                        end_index=chunk.payload.get(
                            "end_char",
                            chunk.start_char + len(chunk.text),
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
                    hash="",
                    name=source_id,
                    document_type=None,  # type: ignore
                    result_score=top_score,
                ),
                chunks=chunk_with_scores,
                top_score=top_score,
                chunk_count=len(chunk_with_scores),
            )
        )

    return result
