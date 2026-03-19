"""LangGraph retrieval pipeline with coverage-driven iteration.

Implements the inner retrieval loop:

    reform_query → split_subqueries → retrieve → check_coverage
                                        ↑              │
                                        └──── retry ───┘ (if uncovered & iteration < max)
                                                       │
                                                    assemble → END

Each pipeline instance is built via ``build_retrieval_pipeline()`` with
configs captured by closures so the LangGraph state stays clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from backend.src.services.conversation.query_service import QueryService
from backend.src.services.conversation.utils.llm_utils import (
    format_source_for_citation,
)
from backend.src.services.retrieval.reranker_service import RerankerService
from backend.src.services.retrieval.vector_service import RetrievedChunk, VectorService

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from backend.src.domain.schemas.config import (
        EmbedderConfig,
        LLMConfig,
        RerankerConfig,
        ScoredRetrieval,
        VectorStoreConfig,
    )


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------


class RetrievalState(TypedDict, total=False):
    """Data flowing through the retrieval pipeline.

    Configs are NOT in the state — they are captured by closures in
    ``build_retrieval_pipeline()``.
    """

    # Input
    user_query: str

    # Pipeline data
    reformed_query: str
    sub_queries: list[str]
    pending_queries: list[str]  # sub-queries not yet covered
    all_chunks: list[Any]  # list[RetrievedChunk] kept in-memory
    context: str  # plain text for coverage check
    formatted_context: str  # XML with chunk IDs for agent citations
    iteration: int


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _chunk_to_scored(chunk: RetrievedChunk, rank: int) -> ScoredRetrieval:
    """Convert a ``RetrievedChunk`` into ``ScoredRetrieval`` for citation
    formatting."""
    from backend.src.domain.schemas.config import ScoredRetrieval as _SR

    return _SR(
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


def _format_chunks_as_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as XML with citation IDs for the agent."""
    if not chunks:
        return ""
    scored = [_chunk_to_scored(c, i) for i, c in enumerate(chunks)]
    parts = [format_source_for_citation(s) for s in scored]
    return "Source material:\n<documents>\n" + "\n".join(parts) + "\n</documents>"


def _plain_text_context(chunks: list[RetrievedChunk]) -> str:
    """Concatenate chunk texts for coverage check (no XML)."""
    return "\n\n".join(c.text for c in chunks)


# ------------------------------------------------------------------
# Pipeline builder
# ------------------------------------------------------------------


def build_retrieval_pipeline(
    llm_config: LLMConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
    token_budget: int = 4096,
    max_iterations: int = 3,
) -> CompiledStateGraph:
    """Compile a retrieval pipeline with configs baked into closures.

    A new compiled graph per request is fine — LangGraph compilation is
    microsecond-cheap.
    """

    # ---- Nodes (close over configs) ----

    async def reform_query_node(state: RetrievalState) -> dict:
        reformed = await QueryService.reform_query(
            user_query=state["user_query"],
            llm_config=llm_config,
        )
        logger.debug("Pipeline: reformed query → {}", reformed)
        return {"reformed_query": reformed, "iteration": 0}

    async def split_subqueries_node(state: RetrievalState) -> dict:
        subs = await QueryService.split_to_subqueries(
            user_query=state["reformed_query"],
            llm_config=llm_config,
        )
        logger.debug("Pipeline: split into {} sub-queries", len(subs))
        return {"sub_queries": subs, "pending_queries": subs}

    async def retrieve_node(state: RetrievalState) -> dict:
        """Search for pending queries, deduplicate, rerank."""
        queries = state.get("pending_queries") or [state["reformed_query"]]
        prev_chunks: list[RetrievedChunk] = list(state.get("all_chunks") or [])
        new_chunks: list[RetrievedChunk] = []

        for q in queries:
            try:
                raw = await VectorService.search(
                    config=vector_config,
                    embed_config=embedder_config,
                    query=q,
                )
                if raw:
                    new_chunks.extend(raw)
            except Exception:
                logger.warning("Vector search failed for sub-query: {}", q)

        # Merge with previously retrieved chunks and deduplicate by text
        combined = prev_chunks + new_chunks
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for c in combined:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)

        # Rerank the full set against the reformed query
        if unique:
            reranker = RerankerService(config=reranker_config)
            unique = reranker.rerank(
                query=state["reformed_query"],
                chunks=unique,
            )

        iteration = state.get("iteration", 0) + 1
        plain = _plain_text_context(unique)

        logger.debug(
            "Pipeline: retrieve iteration {} — {} unique chunks", iteration, len(unique)
        )
        return {
            "all_chunks": unique,
            "context": plain,
            "iteration": iteration,
        }

    async def check_coverage_node(state: RetrievalState) -> dict:
        result = await QueryService.check_coverage(
            sub_queries=state["sub_queries"],
            context=state["context"],
            llm_config=llm_config,
        )
        pending = result.uncovered_queries
        logger.debug(
            "Pipeline: coverage {}/{} — {} uncovered",
            result.covered_count,
            result.total_count,
            len(pending),
        )
        return {"pending_queries": pending}

    def should_retry(state: RetrievalState) -> str:
        pending = state.get("pending_queries") or []
        iteration = state.get("iteration", 0)
        if pending and iteration < max_iterations:
            return "retrieve"
        return "assemble"

    async def assemble_node(state: RetrievalState) -> dict:
        chunks: list[RetrievedChunk] = state.get("all_chunks") or []
        formatted = _format_chunks_as_context(chunks)
        plain = _plain_text_context(chunks)
        logger.debug("Pipeline: assembled {} chunks into context", len(chunks))
        return {
            "formatted_context": formatted,
            "context": plain,
        }

    # ---- Graph ----

    graph = StateGraph(RetrievalState)

    graph.add_node("reform_query", reform_query_node)
    graph.add_node("split_subqueries", split_subqueries_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_coverage", check_coverage_node)
    graph.add_node("assemble", assemble_node)

    graph.add_edge(START, "reform_query")
    graph.add_edge("reform_query", "split_subqueries")
    graph.add_edge("split_subqueries", "retrieve")
    graph.add_edge("retrieve", "check_coverage")
    graph.add_conditional_edges("check_coverage", should_retry)
    graph.add_edge("assemble", END)

    return graph.compile()
