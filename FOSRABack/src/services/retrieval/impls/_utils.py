from typing import Any
from FOSRABack.src.api.schemas import SourceResponseDeep
from FOSRABack.src.api.schemas.source_api_schemas import SourceGroupResponse
from FOSRABack.src.domain.enums import FileType
from FOSRABack.src.domain.schemas import (
    ChunkFull,
    RetrievalConfig,
    RetrievedResult,
    SourceFull,
    VectorSearchResult,
)

from loguru import logger

from FOSRABack.src.domain.schemas.source_schemas import ChunkWithScore, SourceGroup
from FOSRABack.src.storage.utils.converters import DomainStruct, domain_to_response


async def transform_vector_results(
    vector_results: list[VectorSearchResult],
) -> list[RetrievedResult]:
    """Transform vector search results to retrieval results.

    Returns:
        List of transformed retrieval results
    """

    logger.debug(f"Transforming {len(vector_results)} vector results")

    results = []

    for idx, vr in enumerate(vector_results):
        payload = vr.payload

        result: RetrievedResult = RetrievedResult(
            query_text=vr.query_text,
            contents=payload.get("chunk_text", ""),
            similarity_score=vr.score,
            source_id=payload.get("source_id", ""),
            chunk_id=payload.get("chunk_id", ""),
            source_name=payload.get("source_title", "Untitled"),
            file_type=payload.get("file_type", FileType.FILE),
            metadata=payload.get("metadata", {}),
            result_rank=idx,
            origin_type="",
        )

        results.append(result)

    return results


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
    """Remove duplicate chunks based on content similarity.

    Args:
        results: Results to deduplicate

    Returns:
        Deduplicated results
    """

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


class SourcesGroup(DomainStruct):
    source_ids: list[dict[str, SourceResponseDeep | ChunkFull]]


def group_by_source(results: list[RetrievedResult]) -> list[SourceGroup]:
    """Group results by source - returns DOMAIN models."""
    source_map: dict[str, dict[str, Any]] = {}

    for result in results:
        source_id = result.source_id

        if source_id not in source_map:
            # ✅ Create domain model (SourceFull, not SourceResponseDeep)
            source_obj = SourceFull(
                source_id=source_id,
                hash=result.metadata.get("source_hash"),
                name=result.source_name or "",
                origin_path=result.metadata.get("origin_path", ""),
                source_summary="",
                summary_embedding="",
                origin_type=result.metadata.get("origin_type"),
                # Note: SourceFull doesn't have result_score, handle differently
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

    # ✅ Build SourceGroup with domain models
    source_groups: list[SourceGroup] = [
        SourceGroup(
            source=data["source"],  # Now SourceFull (domain)
            chunks=data["chunks"],
            top_score=data["top_score"],
            chunk_count=len(data["chunks"]),
        )
        for data in source_map.values()
    ]

    # Sort by top score
    source_groups.sort(key=lambda x: x.top_score, reverse=True)

    return source_groups  # Return domain models    logger.debug(f"source groups obj: {source_groups} \n")

    # Sort by top score
    source_groups.sort(key=lambda x: x.top_score, reverse=True)

    to_response: list[SourceGroupResponse] = [
        domain_to_response(domain_obj=i, response_cls=SourceGroupResponse)
        for i in source_groups
    ]
    return to_response
