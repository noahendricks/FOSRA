from typing import Any
from backend.src.api.schemas import SourceResponseDeep
from backend.src.domain.schemas import (
    ChunkFull,
    RetrievalConfig,
    RetrievedResult,
    SourceFull,
)

from loguru import logger

from backend.src.domain.schemas.source_schemas import ChunkWithScore, SourceGroup


@staticmethod
async def rerank_results(
    query: str,
    results: list[RetrievedResult],
    config: RetrievalConfig,
) -> list[RetrievedResult]:
    """Apply reranking to results.

    Returns:
        Reranked results

    """

    logger.debug("Reranking not yet implemented, returning original order")

    return results


@staticmethod
def deduplicate_results(
    results: list[RetrievedResult],
) -> list[RetrievedResult]:
    logger.debug(f"Deduplicating {len(results)} results")

    seen_content_hashes: set[str] = set()
    deduplicated = []

    for result in results:
        # Simple hash-based deduplication (first 500 chars)
        content_hash = str(hash(result.contents[:500]))

        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            deduplicated.append(result)

    logger.debug(
        f"Deduplication complete: {len(results)} -> {len(deduplicated)} results"
    )
    return deduplicated




def group_by_source(results: list[RetrievedResult]) -> list[SourceGroup]:
    """Group results by source - returns DOMAIN models."""
    source_map: dict[str, dict[str, Any]] = {}

    for result in results:
        source_id = result.source_id

        if source_id not in source_map:
            source_obj = SourceFull(
                source_id=source_id,
                hash=result.metadata.get("source_hash"),
                name=result.source_name or "",
                origin_path=result.metadata.get("origin_path", ""),
                source_summary="",
                summary_embedding="",
                origin_type=result.metadata.get("origin_type"),
            )

            source_map[source_id] = {
                "source": source_obj,
                "chunks": [],
                "top_score": result.similarity_score,
            }

        chunk = ChunkFull(
            chunk_id=result.chunk_id,
            source_id=result.source_id,
            source_hash=result.metadata.get("source_hash", ""),
            start_index=result.metadata.get("start_index", 0),
            end_index=result.metadata.get("end_index", 0),
            token_count=result.metadata.get("token_count", 0),
            text=result.contents,
        )

        chunk_with_score = ChunkWithScore(
            chunk=chunk,
            similarity_score=result.similarity_score,
            reranker_score=result.reranker_score,
        )

        source_map[source_id]["chunks"].append(chunk_with_score)

        # Update top score
        if result.similarity_score > source_map[source_id]["top_score"]:
            source_map[source_id]["top_score"] = result.similarity_score

    source_groups: list[SourceGroup] = [
        SourceGroup(
            source=data["source"],  # Now SourceFull (domain)
            chunks=data["chunks"],
            top_score=data["top_score"],
            chunk_count=len(data["chunks"]),
        )
        for data in source_map.values()
    ]

    source_groups.sort(key=lambda x: x.top_score, reverse=True)

    return source_groups 
