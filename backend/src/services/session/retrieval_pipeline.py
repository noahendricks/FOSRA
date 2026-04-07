"""LangGraph retrieval pipeline with agentic retrieval loop.

Implements the evolved pipeline:

    expand_query → initial_retrieve → agentic_loop → rerank → END

Each pipeline instance is built via ``build_retrieval_pipeline()`` with
configs captured by closures so the LangGraph state stays clean.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from falkordb import FalkorDB
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models.models import (
    FeedbackItem,
    NaiveFeedbackStrategy,
    NaiveFeedbackStrategyParams,
    RelevanceFeedbackInput,
    RelevanceFeedbackQuery,
)

from backend.src.domain.schemas.retrieval import (
    AccumulatedContext,
    AccumulatedItem,
    ChecklistItem,
    QueryExpansion,
    RetrievalTarget,
)
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.graph_service import GraphService
from backend.src.services.retrieval.reranker_service import RerankerService
from backend.src.services.retrieval.vector_service import (
    CHUNKS_COLLECTION,
    RetrievedChunk,
    VectorService,
)
from backend.src.services.session.query_expander import QueryExpander
from backend.src.services.session.subagent import Subagent
from backend.src.services.session.utils.llm_utils import format_source_for_citation

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from backend.src.settings import (
        EmbedderConfig,
        LLMConfig,
        RerankerConfig,
        ScoredRetrieval,
        VectorStoreConfig,
    )


class RetrievalState(BaseModel):
    """Data flowing through the retrieval pipeline.

    Configs are NOT in the state — they are captured by closures in
    ``build_retrieval_pipeline()``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    user_query: str = ""
    chat_history: str | None = None
    query_expansion: QueryExpansion | None = None
    checklist: list[ChecklistItem] = []
    file_ids: set[str] = set()
    accumulated_context: AccumulatedContext
    iteration: int = 0
    formatted_context: str = ""
    context: str = ""


def _chunk_to_scored(chunk: RetrievedChunk, rank: int) -> ScoredRetrieval:
    """Convert a ``RetrievedChunk`` into ``ScoredRetrieval`` for citation."""
    from backend.src.settings import ScoredRetrieval as _SR

    return _SR(
        rank=rank,
        score=chunk.score,
        text=chunk.text,
        doc_title=chunk.payload.get("doc_title", ""),
        chunk_id=chunk.payload.get("chunk_id", str(rank)),
        doc_id=chunk.payload.get("doc_id", chunk.payload.get("source_id", "")),
        page_number=chunk.payload.get("page_number", 0),
        start_index=chunk.start_char,
        end_index=chunk.payload.get("end_char", chunk.start_char + len(chunk.text)),
    )


# def _format_chunks_as_context(chunks: list[RetrievedChunk]) -> str:
#     """Format chunks as XML with citation IDs."""
#     if not chunks:
#         return ""
#     scored = [_chunk_to_scored(c, i) for i, c in enumerate(chunks)]
#     parts = [format_source_for_citation(s) for s in scored]
#     return "Source material:\n<documents>\n" + "\n".join(parts) + "\n</documents>"


def _retrieved_chunk_to_item(chunk: RetrievedChunk) -> AccumulatedItem:
    """Convert RetrievedChunk to AccumulatedItem."""
    return AccumulatedItem(
        file_id=chunk.payload.get("doc_id", chunk.payload.get("source_id", "")),
        path=chunk.payload.get("doc_title", "unknown"),
        line_start=chunk.start_char,
        line_end=chunk.payload.get("end_char", chunk.start_char + len(chunk.text)),
        content=chunk.text,
        source="vector",
        score=chunk.score,
        qdrant_point_id=chunk.payload.get("point_id"),
    )


def build_retrieval_pipeline(
    llm_config: LLMConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
    falkordb_client: FalkorDB | None = None,
    token_budget: int = 4096,
    max_iterations: int = 5,
    chunks_top_k: int = 10,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    feedback_a: float = 0.24,
    feedback_b: float = 1.35,
    feedback_c: float = 0.59,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile a retrieval pipeline with configs baked into closures.

    Args:
        llm_config: LLM configuration for query expansion and subagent
        embedder_config: Embedder configuration for vector search
        vector_config: Vector store configuration
        reranker_config: Optional reranker configuration
        falkordb_client: Optional FalkorDB client for graph retrieval
        token_budget: Maximum tokens in final context
        max_iterations: Maximum agentic loop iterations
        chunks_top_k: Top-K for chunk retrieval
        dense_weight: Weight for dense vector in weighted RRF fusion
        sparse_weight: Weight for sparse vector in weighted RRF fusion

    Returns:
        Compiled LangGraph pipeline
    """

    store = VectorService._get_store(vector_config, embedder_config)

    if not isinstance(store, QdrantClient):
        raise RuntimeError("Vector store must be QdrantClient for evolved pipeline")

    # Create AsyncQdrantClient from the same connection params
    qdrant_config = vector_config.qdrant_config
    if qdrant_config.data_path:
        qdrant_client = AsyncQdrantClient(path=qdrant_config.data_path)
    elif qdrant_config.url:
        qdrant_client = AsyncQdrantClient(url=qdrant_config.url)
    else:
        qdrant_client = AsyncQdrantClient(
            host=qdrant_config.host,
            port=qdrant_config.port,
        )

    graph_service: GraphService | None = None
    if falkordb_client:
        graph_service = GraphService(falkordb_client)

    async def expand_query_node(state: RetrievalState) -> dict[str, Any]:
        """Expand user query into rewritten query + checklist."""
        expansion = await QueryExpander.expand(
            user_query=state.user_query,
            llm_config=llm_config,
            chat_history=state.chat_history,
        )
        logger.debug(
            "Pipeline: expanded query → {} items in checklist",
            len(expansion.checklist),
        )
        return {
            "query_expansion": expansion,
            "checklist": expansion.checklist,
            "iteration": 0,
            "accumulated_context": AccumulatedContext(),
            "file_ids": set(),
        }

    async def initial_retrieve_node(state: RetrievalState) -> dict[str, Any]:
        """Initial retrieval on rewritten query (chunks + auto-merge hierarchical upgrade)."""
        if state.query_expansion:
            query = state.query_expansion.rewritten_query
        else:
            query = state.user_query

        chunks, file_ids, merged_context = await VectorService.retrieve(
            client=qdrant_client,
            embed_config=embedder_config,
            query=query,
            top_k=chunks_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        acc_items = [_retrieved_chunk_to_item(c) for c in chunks]
        accumulated_context = AccumulatedContext(items=acc_items)

        logger.debug(
            "Pipeline: initial retrieve → {} chunks, {} unique files, merged chars: {}",
            len(chunks),
            len(file_ids),
            len(merged_context),
        )

        return {
            "accumulated_context": accumulated_context,
            "file_ids": file_ids,
            "iteration": 1,
            "merged_context": merged_context,
        }

    async def agentic_loop_node(state: RetrievalState) -> dict[str, Any]:
        """Run subagent iteration: assess coverage + plan retrieval."""
        result = await Subagent.assess_and_plan(
            original_query=state.user_query,
            checklist=state.checklist,
            context=state.accumulated_context,
            iteration=state.iteration,
            llm_config=llm_config,
            max_iterations=max_iterations,
        )

        new_items: list[AccumulatedItem] = []

        for rq in result.retrieval_queries:
            if rq.target in (RetrievalTarget.VECTOR, RetrievalTarget.BOTH):
                vector_items = await _execute_vector_retrieval(
                    rq.query,
                    rq.filters,
                    qdrant_client,
                    embedder_config,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                )
                new_items.extend(vector_items)

            if rq.target in (RetrievalTarget.GRAPH, RetrievalTarget.BOTH):
                if graph_service:
                    graph_items = await _execute_graph_retrieval(
                        rq.query, rq.filters, graph_service, embedder_config
                    )
                    new_items.extend(graph_items)

        updated_context = state.accumulated_context.add_items(new_items)

        new_file_ids = {i.file_id for i in new_items}
        updated_file_ids = state.file_ids | new_file_ids

        logger.debug(
            "Pipeline: agentic iteration {} → {} new items, {} total files",
            state.iteration,
            len(new_items),
            len(updated_file_ids),
        )

        return {
            "checklist": result.checklist,
            "accumulated_context": updated_context,
            "file_ids": updated_file_ids,
            "iteration": state.iteration + 1,
            "_all_answered": result.all_answered,
        }

    async def _execute_vector_retrieval(
        query: str,
        filters: Any | None,
        client: AsyncQdrantClient,
        embed_config: EmbedderConfig,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[AccumulatedItem]:
        """Execute vector retrieval and return AccumulatedItems."""
        filter_dict = None
        if filters and filters.file_ids:
            filter_dict = {"doc_ids": filters.file_ids}

        if embed_config.sparse_enabled:
            chunks = await VectorService.weighted_search(
                client=client,
                collection_name=CHUNKS_COLLECTION,
                embed_config=embed_config,
                query=query,
                filters=filter_dict,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                limit=10,
            )
        else:
            chunks = await VectorService.search_collection(
                client=client,
                collection_name=CHUNKS_COLLECTION,
                embed_config=embed_config,
                query=query,
                filters=filter_dict,
                limit=10,
            )

        return [_retrieved_chunk_to_item(c) for c in chunks]

    async def _execute_graph_retrieval(
        query: str,
        filters: Any | None,
        graph_svc: GraphService,
        embed_config: EmbedderConfig,
    ) -> list[AccumulatedItem]:
        """Execute graph retrieval and return AccumulatedItems."""
        items: list[AccumulatedItem] = []

        embedder = EmbedderService()
        embedded = await embedder.embed_query(query, embed_config)
        if not embedded or not embedded.dense:
            return items

        node_types: list[GraphNodeType] | None = None
        if filters and filters.node_type:
            from backend.src.domain.enums import GraphNodeType

            type_map = {
                "function": GraphNodeType.FUNCTION,
                "class": GraphNodeType.CLASS,
                "method": GraphNodeType.METHOD,
            }
            mapped = type_map.get(filters.node_type.lower())
            if mapped:
                node_types = [mapped]

        file_ids = None
        if filters and filters.file_ids:
            file_ids = [fid for fid in filters.file_ids if fid.isdigit()]

        try:
            result = await graph_svc.semantic_search(
                query_embedding=embedded.dense,
                node_types=node_types,
                file_ids=file_ids,
                limit=10,
            )

            for node in result.nodes:
                items.append(node.to_accumulated_item())

        except Exception as e:
            logger.warning("Graph retrieval failed: {}", e)

        return items

    def should_continue_loop(state: RetrievalState) -> str:
        """Decide if agentic loop should continue."""
        all_answered = getattr(state, "_all_answered", False)
        iteration = getattr(state, "iteration", 0)

        if all_answered:
            return "relevance_feedback"

        if iteration >= max_iterations:
            return "relevance_feedback"

        return "agentic_loop"

    async def relevance_feedback_node(state: RetrievalState) -> dict[str, Any]:
        """Apply Qdrant relevance feedback using accumulated items as positive examples."""
        context = state.accumulated_context
        if not context.items:
            return {"accumulated_context": context}

        feedback_items: list[AccumulatedItem] = [
            item for item in context.items if item.qdrant_point_id and item.score > 0
        ]
        if len(feedback_items) < 2:
            logger.debug(
                "Pipeline: relevance feedback skipped ({} items)", len(feedback_items)
            )
            return {"accumulated_context": context}

        embedder = EmbedderService()
        try:
            embedded = await embedder.embed_query(state.user_query, embedder_config)
            if not embedded or not embedded.dense:
                return {"accumulated_context": context}
        except Exception as e:
            logger.warning("Pipeline: embed query failed for relevance feedback: {}", e)
            return {"accumulated_context": context}

        top_items = sorted(feedback_items, key=lambda x: x.score, reverse=True)[:20]
        feedback = [
            FeedbackItem(
                example=str(item.qdrant_point_id),
                score=item.score,
            )
            for item in top_items
        ]

        feedback_query = RelevanceFeedbackQuery(
            relevance_feedback=RelevanceFeedbackInput(
                target=embedded.dense,
                feedback=feedback,
                strategy=NaiveFeedbackStrategy(
                    naive=NaiveFeedbackStrategyParams(
                        a=feedback_a,
                        b=feedback_b,
                        c=feedback_c,
                    )
                ),
            )
        )

        try:
            results = await qdrant_client.query_points(
                collection_name=CHUNKS_COLLECTION,
                query=feedback_query,
                with_payload=True,
                limit=15,
            )
        except Exception as e:
            logger.warning("Pipeline: relevance feedback query failed: {}", e)
            return {"accumulated_context": context}

        if not results.points:
            return {"accumulated_context": context}

        updated_items: list[AccumulatedItem] = []
        for sp in results.points:
            if sp.payload:
                payload = dict(sp.payload)
                payload["point_id"] = str(sp.id)
                item = AccumulatedItem(
                    file_id=payload.get("doc_id", payload.get("source_id", "")),
                    path=payload.get("doc_title", "unknown"),
                    line_start=payload.get("start_char", 0),
                    line_end=payload.get(
                        "end_char",
                        payload.get("start_char", 0) + len(payload.get("text", "")),
                    ),
                    content=payload.get("text", ""),
                    source="vector",
                    score=sp.score,
                    qdrant_point_id=str(sp.id),
                )
                updated_items.append(item)

        existing_keys = {(i.file_id, i.line_start) for i in context.items}
        unique_updates = [
            i for i in updated_items if (i.file_id, i.line_start) not in existing_keys
        ]
        new_context = AccumulatedContext(items=context.items + unique_updates)

        logger.debug(
            "Pipeline: relevance feedback → {} new items (from {} feedback)",
            len(unique_updates),
            len(top_items),
        )

        return {"accumulated_context": new_context}

    async def rerank_node(state: RetrievalState) -> dict[str, Any]:
        """Rerank accumulated context against original query."""
        context = state.accumulated_context

        if not context.items:
            return {"formatted_context": "", "context": ""}

        if reranker_config:
            chunks = [
                RetrievedChunk(
                    text=item.content,
                    token_count=len(item.content.split()),
                    start_char=item.line_start,
                    score=item.score,
                    payload={
                        "doc_id": item.file_id,
                        "doc_title": item.path,
                    },
                )
                for item in context.items
            ]

            reranker = RerankerService(config=reranker_config)

            reranked = reranker.rerank(
                query=state.user_query,
                chunks=chunks,
            )

            reranked_items = []
            for chunk in reranked:
                for item in context.items:
                    if item.content == chunk.text:
                        reranked_items.append(item)
                        break

            context = AccumulatedContext(items=reranked_items)

        formatted = context.to_formatted_context()
        plain = context.to_plain_text()

        logger.debug(
            "Pipeline: reranked {} items into context",
            len(context.items),
        )

        return {
            "formatted_context": formatted,
            "context": plain,
            "accumulated_context": context,
        }

    graph = StateGraph(RetrievalState)  # type: ignore[arg-type]

    _ = graph.add_node("expand_query", expand_query_node)
    _ = graph.add_node("initial_retrieve", initial_retrieve_node)
    _ = graph.add_node("agentic_loop", agentic_loop_node)
    _ = graph.add_node("relevance_feedback", relevance_feedback_node)
    _ = graph.add_node("rerank", rerank_node)

    _ = graph.add_edge(START, "expand_query")
    _ = graph.add_edge("expand_query", "initial_retrieve")
    _ = graph.add_edge("initial_retrieve", "agentic_loop")
    _ = graph.add_conditional_edges("agentic_loop", should_continue_loop)
    _ = graph.add_edge("relevance_feedback", "rerank")
    _ = graph.add_edge("rerank", END)

    return graph.compile()
