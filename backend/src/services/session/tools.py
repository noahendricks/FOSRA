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
    from backend.src.services.session.retrieval_pipeline import (
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
