from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import tool
from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.src.domain.schemas.retrieval import AccumulatedItem

if TYPE_CHECKING:
    from falkordb import FalkorDB

    from backend.src.settings import (
        ChunkerConfig,
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
        """Search the knowledge base for relevant information based on a query.

        Args:
            query: The search query string.
            target: Where to search - vector, graph, or both.

        Returns:
            Formatted context from the knowledge base or "No relevant information found".
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
        """Search the code graph for symbol relationships and file symbols.

        Args:
            query_type: Type of query - callers, callees, call_chain, inheritance, or file_symbols.
            name: The symbol name or file path to search.
            depth: Maximum depth for call_chain and inheritance queries (default: 3).

        Returns:
            Formatted results of the graph query or error message.
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


def _guess_mime(path: Path) -> str:
    """Guess MIME type from file extension."""
    suffix = path.suffix.lower()
    type_map = {
        ".md": "text/markdown",
        ".mdx": "text/markdown",
        ".txt": "text/plain",
        ".rst": "text/x-rst",
        ".html": "text/html",
        ".pdf": "application/pdf",
        ".py": "text/x-python",
        ".go": "text/x-go",
        ".js": "text/javascript",
        ".ts": "text/typescript",
        ".tsx": "text/typescript-jsx",
        ".rs": "text/x-rust",
        ".java": "text/x-java",
        ".cpp": "text/x-c++",
        ".c": "text/x-c",
        ".h": "text/x-c",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".toml": "text/x-toml",
    }
    return type_map.get(suffix, "application/octet-stream")


def create_ingest_codebase_tool(
    session_factory: async_sessionmaker,
    falkordb_client: "FalkorDB",
    embedder_config: "EmbedderConfig",
) -> Any:
    """Tool for ingesting a codebase folder into the graph database."""

    @tool
    async def ingest_codebase_tool(
        directory_path: str,
        repo_name: str | None = None,
    ) -> str:
        """Ingest a codebase directory into the knowledge graph.

        Use this tool when the user wants to index or re-index a codebase directory
        so that code symbols, call relationships, and file structure can be searched.

        Args:
            directory_path: Absolute path to the codebase directory to index.
            repo_name: Optional name for this repository (defaults to folder name).

        Returns:
            Status message with ingestion statistics or error details.
        """
        from backend.src.tasks.codebase_ingestion import ingest_codebase

        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory not found: {directory_path}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory_path}"

        repo = repo_name or path.name
        logger.info("Ingesting codebase: {} (repo: {})", directory_path, repo)

        try:
            result = await ingest_codebase(
                directory_path=str(path.absolute()),
                repo_name=repo,
                embedder_config=embedder_config,
                falkordb_client=falkordb_client,
                session_factory=session_factory,
                recursive=True,
            )
            files_processed = result.get("files_processed", 0)
            total_nodes = result.get("total_nodes", 0)
            total_edges = result.get("total_call_edges", 0)
            errors = result.get("errors", [])
            error_info = f" ({len(errors)} errors)" if errors else ""
            return (
                f"Codebase ingestion complete for {directory_path} (repo: {repo}). "
                f"Processed {files_processed} files, {total_nodes} nodes, "
                f"{total_edges} call edges.{error_info}"
            )
        except Exception as e:
            logger.error("Codebase ingestion failed: {}", e)
            return f"Error ingesting codebase: {e}"

    return ingest_codebase_tool


def create_ingest_file_tool(
    session_factory: async_sessionmaker,
    embedder_config: "EmbedderConfig",
    vector_config: "VectorStoreConfig",
    chunker_config: "ChunkerConfig",
) -> Any:
    """Tool for ingesting files into the vector database."""

    @tool
    async def ingest_file_tool(
        file_path: str,
        source_type: Literal["doc", "code"] = "doc",
    ) -> str:
        """Ingest a file into the knowledge base for retrieval.

        Use this tool when the user wants to index a document or code file
        so its contents can be searched and referenced.

        Args:
            file_path: Absolute path to the file to index.
            source_type: "doc" for documents (markdown, pdf, text), "code" for code files.

        Returns:
            Status message with indexing results or error details.
        """
        from backend.src.domain.enums import FileSourceType
        from backend.src.domain.schemas.doc import Doc
        from backend.src.services.processing.docling_loader import DoclingLoader
        from backend.src.tasks.doc_ingestion import ingest_docs

        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        if not path.is_file():
            return f"Error: Not a file: {file_path}"

        mime = _guess_mime(path)
        try:
            doc = DoclingLoader.parse_file_sync(str(path.absolute()), mime_type=mime)
        except Exception as e:
            logger.error("Failed to parse file {}: {}", file_path, e)
            return f"Error parsing file: {e}"

        doc.metadata.source_type = (
            FileSourceType.DOC.value
            if source_type == "doc"
            else FileSourceType.CODEBASE.value
        )
        logger.info("Ingesting file: {} (type: {})", file_path, source_type)

        try:
            result = await ingest_docs(
                docs=[doc.to_dict()],
                chunker_config=chunker_config,
                embedder_config=embedder_config,
                vector_config=vector_config,
                session_factory=session_factory,
            )
            chunks = result.get("chunks_upserted", 0)
            docs_proc = result.get("docs_processed", 0)
            return (
                f"File ingestion complete for {file_path} (type: {source_type}). "
                f"Processed {docs_proc} docs, {chunks} chunks upserted."
            )
        except Exception as e:
            logger.error("File ingestion failed: {}", e)
            return f"Error ingesting file: {e}"

    return ingest_file_tool
