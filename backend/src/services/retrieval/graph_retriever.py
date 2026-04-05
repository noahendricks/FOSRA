"""Graph retrieval service for FalkorDB.

Separates retrieval (read) concerns from construction (write) concerns in GraphService.
Provides semantic vector search and structural traversals on the code graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.src.domain.enums import GraphNodeType
from backend.src.domain.schemas.graph import GraphQueryResult

if TYPE_CHECKING:
    from falkordb import Graph

    from backend.src.services.retrieval.graph_service import GraphService


class GraphRetriever:
    """Retrieves code context from FalkorDB graph.

    Provides two retrieval modes:
    - Semantic: vector similarity search on embedded Function/Class nodes
    - Structural: Cypher traversals for callers, callees, inheritance, etc.
    """

    def __init__(self, graph_service: GraphService):
        self._graph_service = graph_service

    def _get_graph(self) -> Graph:
        return self._graph_service._get_graph()

    async def semantic_search(
        self,
        query_embedding: list[float],
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> GraphQueryResult:
        """Search for nodes by embedding similarity.

        Args:
            query_embedding: The query vector to search with
            node_types: Filter to specific node types (Function, Class, Method)
            file_ids: Optional list of file IDs to filter results
            limit: Maximum number of results

        Returns:
            GraphQueryResult with matching nodes sorted by similarity
        """
        return await self._graph_service.semantic_search(
            query_embedding=query_embedding,
            node_types=node_types,
            file_ids=file_ids,
            limit=limit,
        )

    def get_callers(
        self, name: str, depth: int = 1, limit: int = 50
    ) -> GraphQueryResult:
        """Find functions that call the given function.

        Args:
            name: Function name or qualified name
            depth: How many levels of callers to traverse (1 = direct callers only)
            limit: Maximum number of results

        Returns:
            GraphQueryResult with calling functions
        """
        return self._graph_service.structural_query(
            query_type="callers",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_callees(self, name: str, limit: int = 50) -> GraphQueryResult:
        """Find functions called by the given function.

        Args:
            name: Function name or qualified name
            limit: Maximum number of results

        Returns:
            GraphQueryResult with called functions
        """
        return self._graph_service.structural_query(
            query_type="callees",
            name=name,
            limit=limit,
        )

    def get_call_chain(
        self, name: str, depth: int = 5, limit: int = 50
    ) -> GraphQueryResult:
        """Get full outbound call chain from a function.

        Args:
            name: Function name to start from
            depth: Maximum traversal depth
            limit: Maximum number of paths

        Returns:
            GraphQueryResult with call chain paths
        """
        return self._graph_service.structural_query(
            query_type="call_chain",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_file_symbols(self, file_id: str, limit: int = 100) -> GraphQueryResult:
        """Get all symbols (functions, classes) defined in a file.

        Args:
            file_id: The file's doc_id
            limit: Maximum number of results

        Returns:
            GraphQueryResult with file's symbols
        """
        graph = self._get_graph()

        query = """
        MATCH (file:File {file_id: $file_id})-[:CONTAINS]->(node)
        RETURN node
        LIMIT $limit
        """
        result = graph.query(query, params={"file_id": file_id, "limit": limit})

        nodes = []
        for row in result.result_set:
            if row:
                nodes.append(self._graph_service._node_to_code_node(row[0]))

        return GraphQueryResult(
            nodes=nodes,
            total_count=len(nodes),
            query_type="structural",
        )

    def get_inheritance_chain(
        self, name: str, depth: int = 10, limit: int = 50
    ) -> GraphQueryResult:
        """Get inheritance chain for a class.

        Args:
            name: Class name
            depth: Maximum traversal depth
            limit: Maximum number of paths

        Returns:
            GraphQueryResult with inheritance paths
        """
        return self._graph_service.structural_query(
            query_type="inheritance",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_file_imports(self, file_id: str, limit: int = 100) -> GraphQueryResult:
        """Get files imported by a given file.

        Args:
            file_id: The file's doc_id
            limit: Maximum number of results

        Returns:
            GraphQueryResult with imported files
        """
        graph = self._get_graph()

        query = """
        MATCH (f:File {file_id: $file_id})-[:IMPORTS]->(imported:File)
        RETURN imported
        LIMIT $limit
        """
        result = graph.query(query, params={"file_id": file_id, "limit": limit})

        nodes = []
        for row in result.result_set:
            if row:
                nodes.append(self._graph_service._node_to_code_node(row[0]))

        return GraphQueryResult(
            nodes=nodes,
            total_count=len(nodes),
            query_type="structural",
        )

    def search_by_name(
        self, name: str, node_type: str = "Function", limit: int = 20
    ) -> GraphQueryResult:
        """Search for nodes by name using full-text index.

        Args:
            name: Name to search for (supports partial matches)
            node_type: Node label to search (Function, Class)
            limit: Maximum number of results

        Returns:
            GraphQueryResult with matching nodes
        """
        graph = self._get_graph()

        query = f"""
        CALL db.idx.fulltext.queryNodes('{node_type}', $name)
        YIELD node
        RETURN node
        LIMIT $limit
        """
        result = graph.query(query, params={"name": f"*{name}*", "limit": limit})

        nodes = []
        for row in result.result_set:
            if row:
                nodes.append(self._graph_service._node_to_code_node(row[0]))

        return GraphQueryResult(
            nodes=nodes,
            total_count=len(nodes),
            query_type="structural",
        )
