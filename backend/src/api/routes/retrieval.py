"""Retrieval API endpoints for direct vector/graph search.

Provides REST endpoint for direct retrieval without going through
the agent tool interface.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger

from backend.src.api.dependencies import get_infra
from backend.src.api.lifecycle import Infrastructure
from backend.src.api.schemas.base import BaseModelFlex
from backend.src.domain.enums import VectorStoreType
from backend.src.domain.schemas.retrieval import (
    AccumulatedItem,
    RetrievalFilters,
    RetrievalTarget,
)
from backend.src.settings import (
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.settings.config import QdrantConfig
from backend.src.settings import settings


router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class RetrievalSearchRequest(BaseModelFlex):
    """Request body for POST /retrieval/search."""

    query: str
    target: RetrievalTarget = RetrievalTarget.BOTH
    filters: RetrievalFilters | None = None
    top_k: int = 10
    token_budget: int = 4096
    merge_threshold: float = 0.5
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


class RetrievalSearchResponse(BaseModelFlex):
    """Response body for POST /retrieval/search."""

    chunks: list[AccumulatedItem]
    file_ids: list[str]
    merged_context: str
    query_expansion: dict[str, Any] | None = None


async def get_embedder_config() -> EmbedderConfig:
    """Get embedder config from settings."""
    return EmbedderConfig(
        embedder_type=settings.embedding.model_type,
        dense_model=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
        sparse_enabled=True,
        dense_dimensions=1024,
    )


async def get_vector_config() -> VectorStoreConfig:
    """Get vector store config from settings."""
    return VectorStoreConfig(
        preferred_store=VectorStoreType.QDRANT,
        qdrant_config=QdrantConfig(
            collection_name=settings.vectors.collection_name,
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            url=settings.qdrant.url,
            data_path=settings.qdrant.data_path,
        ),
    )


def _retrieved_chunk_to_accumulated_item(
    chunk: Any, source: str = "vector"
) -> AccumulatedItem:
    """Convert a RetrievedChunk to an AccumulatedItem."""
    payload = chunk.payload or {}
    return AccumulatedItem(
        file_id=payload.get("doc_id", ""),
        path=payload.get("doc_title", ""),
        line_start=0,
        line_end=0,
        content=chunk.text,
        source=source,
        score=chunk.score,
        node_type=None,
        qdrant_point_id=payload.get("point_id"),
    )


def _graph_node_to_accumulated_item(node: Any, score: float) -> AccumulatedItem:
    """Convert a CodeNode to an AccumulatedItem."""
    content_parts = []
    if node.signature:
        content_parts.append(node.signature)
    if node.docstring:
        content_parts.append(node.docstring)
    if node.source_code:
        content_parts.append(node.source_code)
    content = "\n\n".join(content_parts) if content_parts else node.name

    return AccumulatedItem(
        file_id=node.file_id,
        path=node.file_path,
        line_start=node.line_start,
        line_end=node.line_end,
        content=content,
        source="graph",
        score=score,
        node_type=node.node_type.value
        if hasattr(node.node_type, "value")
        else str(node.node_type),
    )


def _fuse_results_rrf(
    vector_items: list[AccumulatedItem],
    graph_items: list[AccumulatedItem],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10,
) -> list[AccumulatedItem]:
    """Fuse vector and graph results using weighted RRF."""
    RRF_K = 60
    chunk_scores: dict[str, tuple[AccumulatedItem, float]] = {}

    for rank, item in enumerate(vector_items):
        key = f"{item.file_id}:{item.content[:50]}"
        score = (1.0 / (RRF_K + rank)) * dense_weight
        if key in chunk_scores:
            chunk_scores[key] = (chunk_scores[key][0], chunk_scores[key][1] + score)
        else:
            chunk_scores[key] = (item, score)

    for rank, item in enumerate(graph_items):
        key = f"{item.file_id}:{item.content[:50]}"
        score = (1.0 / (RRF_K + rank)) * sparse_weight
        if key in chunk_scores:
            chunk_scores[key] = (chunk_scores[key][0], chunk_scores[key][1] + score)
        else:
            chunk_scores[key] = (item, score)

    fused = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in fused[:top_k]]


@router.post("/search")
async def search_retrieval(
    request: Annotated[RetrievalSearchRequest, Body()],
    infra: Annotated[Infrastructure, Depends(get_infra)],
    embedder_config: Annotated[EmbedderConfig, Depends(get_embedder_config)],
    vector_config: Annotated[VectorStoreConfig, Depends(get_vector_config)],
) -> RetrievalSearchResponse:
    """Direct retrieval search endpoint.

    Allows direct vector/graph retrieval without going through the agent tool interface.
    Supports vector-only, graph-only, or hybrid (both) retrieval with RRF fusion.

    Args:
        request: Search parameters including query, target, filters, and ranking options
        infra: Infrastructure singleton with Qdrant and FalkorDB clients
        embedder_config: Embedder configuration
        vector_config: Vector store configuration

    Returns:
        Retrieved chunks, unique file IDs, and merged context
    """
    from backend.src.services.retrieval.vector_service import VectorService
    from backend.src.services.retrieval.graph_service import GraphService
    from backend.src.services.processing.embedder_service import EmbedderService

    if infra.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    # Build filters dict for vector search
    filters_dict: dict[str, Any] | None = None
    if request.filters:
        filters_dict = {}
        if request.filters.file_ids:
            filters_dict["doc_ids"] = request.filters.file_ids

    if request.target == RetrievalTarget.VECTOR:
        chunks, file_ids, merged_context = await VectorService.retrieve(
            client=infra.qdrant_client,
            embed_config=embedder_config,
            query=request.query,
            filters=filters_dict,
            top_k=request.top_k,
            token_budget=request.token_budget,
            merge_threshold=request.merge_threshold,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
        )
        items = [_retrieved_chunk_to_accumulated_item(c) for c in chunks]

    elif request.target == RetrievalTarget.GRAPH:
        if infra.falkordb_client is None:
            raise HTTPException(status_code=503, detail="FalkorDB not available")

        graph_service = GraphService(
            client=infra.falkordb_client,
            graph_name=settings.falkordb.graph_name,
        )

        embedded = await EmbedderService().embed_query(
            request.query, config=embedder_config
        )
        if not embedded or not embedded.dense:
            raise HTTPException(status_code=500, detail="Query embedding failed")

        node_types = None
        if request.filters and request.filters.node_type:
            from backend.src.domain.enums import GraphNodeType

            try:
                node_types = [GraphNodeType(request.filters.node_type)]
            except ValueError:
                pass

        file_ids_list = None
        if request.filters and request.filters.file_ids:
            file_ids_list = [
                int(fid) for fid in request.filters.file_ids if fid.isdigit()
            ]

        graph_result = await graph_service.semantic_search(
            query_embedding=embedded.dense,
            node_types=node_types,
            file_ids=file_ids_list,
            limit=request.top_k,
        )

        items = [
            _graph_node_to_accumulated_item(node, score)
            for node, score in zip(
                graph_result.nodes, [r.score for r in graph_result.nodes]
            )
        ]
        file_ids = list({item.file_id for item in items})
        merged_context = "\n\n".join(item.content for item in items)

    else:
        vector_task = VectorService.retrieve(
            client=infra.qdrant_client,
            embed_config=embedder_config,
            query=request.query,
            filters=filters_dict,
            top_k=request.top_k,
            token_budget=request.token_budget,
            merge_threshold=request.merge_threshold,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
        )

        if infra.falkordb_client is None:
            raise HTTPException(
                status_code=503, detail="FalkorDB not available for graph search"
            )

        graph_service = GraphService(
            client=infra.falkordb_client,
            graph_name=settings.falkordb.graph_name,
        )

        embedded = await EmbedderService().embed_query(
            request.query, config=embedder_config
        )
        if not embedded or not embedded.dense:
            raise HTTPException(status_code=500, detail="Query embedding failed")

        node_types = None
        if request.filters and request.filters.node_type:
            from backend.src.domain.enums import GraphNodeType

            try:
                node_types = [GraphNodeType(request.filters.node_type)]
            except ValueError:
                pass

        file_ids_list = None
        if request.filters and request.filters.file_ids:
            file_ids_list = [
                int(fid) for fid in request.filters.file_ids if fid.isdigit()
            ]

        graph_task = graph_service.semantic_search(
            query_embedding=embedded.dense,
            node_types=node_types,
            file_ids=file_ids_list,
            limit=request.top_k,
        )

        vector_results, graph_result = await vector_task, graph_task

        vector_chunks, vector_file_ids, vector_merged = vector_results
        vector_items = [_retrieved_chunk_to_accumulated_item(c) for c in vector_chunks]

        graph_items = [
            _graph_node_to_accumulated_item(node, score)
            for node, score in zip(
                graph_result.nodes, [r.score for r in graph_result.nodes]
            )
        ]

        items = _fuse_results_rrf(
            vector_items,
            graph_items,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
            top_k=request.top_k,
        )
        file_ids = list({item.file_id for item in items})
        merged_context = "\n\n".join(item.content for item in items)

    return RetrievalSearchResponse(
        chunks=items,
        file_ids=file_ids,
        merged_context=merged_context,
        query_expansion=None,
    )
