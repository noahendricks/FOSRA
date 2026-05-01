"""Graph retrieval service for FalkorDB.

Separates retrieval (read) concerns from construction (write) concerns in GraphService.
Provides semantic vector search and structural traversals on the code graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.src.domain.schemas.graph import CallEdge, GraphQueryResult
from backend.src.domain.schemas.graph_types import GraphNodeType

if TYPE_CHECKING:
    from falkordb import Graph

    from backend.src.services.retrieval.graph_service import GraphService


class GraphRetriever:
    """retrieves code context from FalkorDB graph."""

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
        return await self._graph_service.semantic_search(
            query_embedding=query_embedding,
            node_types=node_types,
            file_ids=file_ids,
            limit=limit,
        )

    def get_callers(
        self, name: str, depth: int = 1, limit: int = 50
    ) -> GraphQueryResult:
        return self._graph_service.structural_query(
            query_type="callers",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_callees(self, name: str, limit: int = 50) -> GraphQueryResult:
        return self._graph_service.structural_query(
            query_type="callees",
            name=name,
            limit=limit,
        )

    def get_call_chain(
        self, name: str, depth: int = 5, limit: int = 50
    ) -> GraphQueryResult:
        return self._graph_service.structural_query(
            query_type="call_chain",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_file_symbols(self, file_id: str, limit: int = 100) -> GraphQueryResult:
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
        return self._graph_service.structural_query(
            query_type="inheritance",
            name=name,
            depth=depth,
            limit=limit,
        )

    def get_file_imports(self, file_id: str, limit: int = 100) -> GraphQueryResult:
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
