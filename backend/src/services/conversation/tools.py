"""Tools for the FOSRA DeepAgent.

Provides a factory to create ``search_knowledge_base`` tool that wraps
the LangGraph retrieval pipeline. The factory captures user-specific
configs via closures so each tool instance is request-scoped.

Supports both vector (document) and graph (code structure) retrieval.

Built-in deepagents middleware already provides ``read_file``, ``ls``,
``glob``, and ``grep`` — so we don't create those here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool
from loguru import logger

from backend.src.domain.schemas.retrieval import AccumulatedItem

if TYPE_CHECKING:
    from falkordb import FalkorDB

    from backend.src.settings import (
        EmbedderConfig,
        LLMConfig,
        RerankerConfig,
        VectorStoreConfig,
    )


@dataclass
class RetrievalResultStore:
    """Mutable container populated by retrieval tools so the caller
    (workspace route) can access the retrieved items for building
    source-group SSE events.
    """

    items: list[AccumulatedItem] = field(default_factory=list)


def create_retrieval_tool(
    llm_config: LLMConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    reranker_config: RerankerConfig | None = None,
    falkordb_client: "FalkorDB | None" = None,
    token_budget: int = 4096,
    max_iterations: int = 5,
    result_store: RetrievalResultStore | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    feedback_a: float = 0.24,
    feedback_b: float = 1.35,
    feedback_c: float = 0.59,
):
    """Create a ``search_knowledge_base`` tool bound to the given configs.

    Parameters
    ----------
    falkordb_client:
        Optional FalkorDB client for graph-based code retrieval.
    result_store:
        If provided, the tool writes its retrieved items into this
        store after each invocation so the caller can forward them to
        the frontend.
    rrf_parent_weight:
        Weight for parent results in weighted RRF fusion (default 3.0).
    rrf_chunk_weight:
        Weight for chunk results in weighted RRF fusion (default 1.0).
    feedback_a, feedback_b, feedback_c:
        Naive relevance feedback formula parameters (default 0.24, 1.35, 0.59).
    """
    from backend.src.services.conversation.retrieval_pipeline import (
        build_retrieval_pipeline,
    )

    pipeline = build_retrieval_pipeline(
        llm_config=llm_config,
        embedder_config=embedder_config,
        vector_config=vector_config,
        reranker_config=reranker_config,
        falkordb_client=falkordb_client,
        token_budget=token_budget,
        max_iterations=max_iterations,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        feedback_a=feedback_a,
        feedback_b=feedback_b,
        feedback_c=feedback_c,
    )

    @tool
    async def search_knowledge_base(
        query: str,
        target: Literal["vector", "graph", "both"] = "both",
    ) -> str:
        """Search the user's personal knowledge base for information.

        Use this tool for ANY question that requires looking up documents,
        notes, code, or previously ingested content. Supports both vector
        (documents) and graph (code structure) retrieval.

        Args:
            query: Natural language question. Be specific and include key
                   terms, function names, or file names when known.
            target: Where to search:
                - "vector": Search documents and notes (Qdrant)
                - "graph": Search code structure and relationships (FalkorDB)
                - "both": Search both sources (default)

        Returns:
            Formatted context from the knowledge base with citation IDs.
            Use citation IDs in ``[citation:N]`` format when referencing.
        """
        logger.info("Retrieval tool invoked: query='{}', target={}", query, target)

        result = await pipeline.ainvoke({"user_query": query})

        context = result.get("accumulated_context")
        if context and context.items:
            if result_store is not None:
                result_store.items.extend(context.items)

            formatted = context.to_formatted_context()
            logger.info(
                "Retrieval tool returning {} items from knowledge base",
                len(context.items),
            )
            return formatted

        return "No relevant information was found in the knowledge base."

    return search_knowledge_base


def create_graph_tool(
    falkordb_client: "FalkorDB",
    embedder_config: "EmbedderConfig",
    result_store: RetrievalResultStore | None = None,
):
    """Create a ``search_code_graph`` tool for structural code queries.

    This tool is for querying code structure relationships:
    - Who calls a function?
    - What does a function call?
    - Inheritance chains
    - File imports

    Args:
        falkordb_client: FalkorDB client for graph queries
        embedder_config: Embedder config for semantic graph search
        result_store: Optional store for retrieved items
    """
    from backend.src.services.retrieval.graph_retriever import GraphRetriever
    from backend.src.services.retrieval.graph_service import GraphService

    graph_service = GraphService(falkordb_client)
    retriever = GraphRetriever(graph_service)

    @tool
    async def search_code_graph(
        query_type: Literal[
            "callers", "callees", "call_chain", "inheritance", "file_symbols"
        ],
        name: str,
        depth: int = 3,
    ) -> str:
        """Query the code structure graph for relationships.

        Use this tool to understand code structure and dependencies:
        - Find who calls a function ("callers")
        - Find what a function calls ("callees")
        - Trace full call chains ("call_chain")
        - Find class inheritance ("inheritance")
        - Get all symbols in a file ("file_symbols")

        Args:
            query_type: Type of structural query to run
            name: Function or class name to query
            depth: Maximum traversal depth (default 3)

        Returns:
            Formatted results showing code relationships.
        """
        logger.info(
            "Graph tool invoked: type={}, name={}, depth={}",
            query_type,
            name,
            depth,
        )

        try:
            match query_type:
                case "callers":
                    result = retriever.get_callers(name, depth=depth)
                case "callees":
                    result = retriever.get_callees(name)
                case "call_chain":
                    result = retriever.get_call_chain(name, depth=depth)
                case "inheritance":
                    result = retriever.get_inheritance_chain(name, depth=depth)
                case "file_symbols":
                    result = retriever.get_file_symbols(name)
                case _:
                    return f"Unknown query type: {query_type}"

            if not result.nodes and not result.paths:
                return f"No results found for {query_type} query on '{name}'"

            lines = []
            if result.nodes:
                lines.append(f"Found {len(result.nodes)} nodes:")
                for node in result.nodes[:10]:
                    lines.append(
                        f"  - {node.node_type.value} {node.name} "
                        f"({node.file_path}:{node.line_start})"
                    )

            if result.paths:
                lines.append(f"\nFound {len(result.paths)} paths:")
                for i, path in enumerate(result.paths[:5]):
                    path_str = " → ".join(n.name for n in path)
                    lines.append(f"  Path {i + 1}: {path_str}")

            if result_store:
                for node in result.nodes:
                    result_store.items.append(node.to_accumulated_item())

            return "\n".join(lines)

        except Exception as e:
            logger.error("Graph query failed: {}", e)
            return f"Graph query failed: {e}"

    return search_code_graph
