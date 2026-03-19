"""Tools for the FOSRA DeepAgent.

Provides a factory to create a ``search_knowledge_base`` tool that wraps
the LangGraph retrieval pipeline.  The factory captures user-specific
configs via closures so each tool instance is request-scoped.

Built-in deepagents middleware already provides ``read_file``, ``ls``,
``glob``, and ``grep`` — so we don't create those here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from loguru import logger

if TYPE_CHECKING:
    from backend.src.domain.schemas.config import (
        EmbedderConfig,
        LLMConfig,
        RerankerConfig,
        VectorStoreConfig,
    )
    from backend.src.services.retrieval.vector_service import RetrievedChunk


# ------------------------------------------------------------------
# Side-channel for source data
# ------------------------------------------------------------------


@dataclass
class RetrievalResultStore:
    """Mutable container populated by the retrieval tool so the caller
    (workspace route) can access the retrieved chunks for building
    source-group SSE events.
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)


# ------------------------------------------------------------------
# Tool factory
# ------------------------------------------------------------------


def create_retrieval_tool(
    llm_config: LLMConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
    token_budget: int = 4096,
    max_iterations: int = 3,
    result_store: RetrievalResultStore | None = None,
):
    """Create a ``search_knowledge_base`` tool bound to the given configs.

    Parameters
    ----------
    result_store:
        If provided, the tool writes its retrieved chunks into this
        store after each invocation so the caller can forward them to
        the frontend.
    """
    from backend.src.services.conversation.retrieval_pipeline import (
        build_retrieval_pipeline,
    )

    pipeline = build_retrieval_pipeline(
        llm_config=llm_config,
        embedder_config=embedder_config,
        vector_config=vector_config,
        reranker_config=reranker_config,
        token_budget=token_budget,
        max_iterations=max_iterations,
    )

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Search the user's personal knowledge base for information relevant to the query.

        Use this tool for ANY question that requires looking up documents,
        notes, code, or previously ingested content from the user's
        knowledge base.  The tool runs a full retrieval pipeline
        internally: query reformulation, multi-query search, cross-encoder
        reranking, and coverage-driven iteration.

        Args:
            query: Natural language question.  Be specific and include key
                   terms, function names, or file names when known.

        Returns:
            Formatted context from the knowledge base with citation IDs.
            Use the chunk IDs in ``[citation:chunk_id]`` format when
            referencing the returned information.
        """
        logger.info("Retrieval tool invoked with query: {}", query)
        result = await pipeline.ainvoke({"user_query": query})

        chunks = result.get("all_chunks") or []
        if result_store is not None:
            result_store.chunks.extend(chunks)

        formatted = result.get("formatted_context", "")
        if not formatted:
            return "No relevant information was found in the knowledge base."

        logger.info("Retrieval tool returning context from {} chunks", len(chunks))
        return formatted

    return search_knowledge_base
