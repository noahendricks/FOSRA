from __future__ import annotations

from typing import TYPE_CHECKING, Any

from falkordb import FalkorDB, Node
from loguru import logger

if TYPE_CHECKING:
    from falkordb import Graph
    from backend.src.domain.schemas.config import EmbedderConfig

from backend.src.domain.enums import GraphNodeType
from backend.src.domain.schemas.doc import Chunk, ChunkMetadata
from backend.src.domain.schemas.graph import (
    CallEdge,
    CodeNode,
    GraphQueryResult,
    GraphResult,
    InheritanceEdge,
    ResolvedImport,
)
from backend.src.services.processing.callgraph_service import CallGraphService
from backend.src.services.processing.embedder_service import EmbedderService


class GraphService:
    _graphs: dict[str, Graph] = {}

    def __init__(self, client: FalkorDB, graph_name: str = "codebase"):
        self._client = client
        self._graph_name = graph_name

    def _get_graph(self) -> Graph:
        if self._graph_name not in self._graphs:
            self._graphs[self._graph_name] = self._client.select_graph(self._graph_name)
        return self._graphs[self._graph_name]

    async def upsert_file_graph(
        self,
        graph_result: GraphResult,
        embedder_config: "EmbedderConfig",
    ) -> dict[str, Any]:
        """
        upsert nodes and edges from a file's graph extraction into falkordb.
        embeds function/class nodes for semantic search.
        """
        graph = self._get_graph()
        stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "nodes_updated": 0,
        }

        node_embeddings = await self._embed_nodes(graph_result.nodes, embedder_config)

        for node in graph_result.nodes:
            self._upsert_node(graph, node, node_embeddings.get(node.qualified_name))
            stats["nodes_created"] += 1

        for edge in graph_result.call_edges:
            self._upsert_call_edge(graph, edge)
            stats["edges_created"] += 1

        for edge in graph_result.inheritance_edges:
            self._upsert_inheritance_edge(graph, edge)
            stats["edges_created"] += 1

        for imp in graph_result.imports:
            if imp.target_file_id:
                self._upsert_import_edge(graph, imp)
                stats["edges_created"] += 1

        logger.info(
            f"Upserted file {graph_result.file_path}: "
            f"{stats['nodes_created']} nodes, {stats['edges_created']} edges"
        )
        return stats

    async def _embed_nodes(
        self,
        nodes: list[CodeNode],
        embedder_config: "EmbedderConfig",
    ) -> dict[str, list[float]]:
        """
        generate embeddings for function and class nodes.
        embedding input: signature + docstring + first N lines of body.
        """
        embeddings = {}
        embedder = EmbedderService()

        nodes_to_embed = [
            n
            for n in nodes
            if n.node_type
            in (GraphNodeType.FUNCTION, GraphNodeType.METHOD, GraphNodeType.CLASS)
        ]

        if not nodes_to_embed:
            return embeddings

        chunks = []

        for node in nodes_to_embed:
            text = self._build_embedding_text(node)
            chunk = Chunk(
                text=text,
                metadata=ChunkMetadata(
                    doc_id=str(node.file_id),
                    chunk_id=node.qualified_name,
                ),
            )
            chunks.append(chunk)

        embedded_chunks = await embedder.embed_chunks(chunks, embedder_config)

        for node, chunk in zip(nodes_to_embed, embedded_chunks):
            if chunk.metadata.dense_embedding:
                embeddings[node.qualified_name] = chunk.metadata.dense_embedding

        return embeddings

    def _build_embedding_text(self, node: CodeNode) -> str:
        """build the text to embed for a code node."""
        parts = []

        if node.signature:
            sig_str = self._signature_to_string(node.signature, node.name)
            parts.append(sig_str)

        if node.docstring:
            parts.append(node.docstring)

        if node.source_code:
            lines = node.source_code.split("\n")
            max_lines = min(20, len(lines))
            body_lines = lines[:max_lines]
            parts.append("\n".join(body_lines))

        return "\n\n".join(parts)

    def _signature_to_string(self, sig: Any, name: str) -> str:
        """convert a signature object to a string representation."""
        decorators = ""
        if sig.decorators:
            decorators = "".join(f"{d}\n" for d in sig.decorators)

        async_kw = "async " if sig.is_async else ""

        params = []
        for p in sig.parameters:
            param_str = p.name
            if p.type_annotation:
                param_str += f": {p.type_annotation}"
            if p.default_value:
                param_str += f" = {p.default_value}"
            if p.is_variadic:
                param_str = f"*{param_str}"
            elif p.is_keyword:
                param_str = f"**{param_str}"
            params.append(param_str)

        params_str = ", ".join(params)

        receiver = ""
        if sig.receiver:
            receiver = f"({sig.receiver}) "

        return_str = ""
        if sig.return_type:
            return_str = f" -> {sig.return_type}"

        return f"{decorators}{async_kw}def {receiver}{name}({params_str}){return_str}:"

    def _upsert_node(
        self,
        graph: "Graph",
        node: CodeNode,
        embedding: list[float] | None,
    ) -> None:
        """create or update a node in the graph."""
        label = node.node_type.value

        props = {
            "file_id": node.file_id,
            "name": node.name,
            "qualified_name": node.qualified_name,
            "file_path": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end,
        }

        if node.docstring:
            props["docstring"] = node.docstring

        if node.signature:
            props["signature"] = self._signature_to_string(node.signature, node.name)
            props["is_async"] = node.signature.is_async

        if embedding:
            props["embedding"] = embedding

        query = f"""
        MERGE (n:{label} {{qualified_name: $qualified_name}})
        SET n += $props
        RETURN n
        """
        graph.query(
            query, params={"qualified_name": node.qualified_name, "props": props}
        )

    def _upsert_call_edge(self, graph: "Graph", edge: CallEdge) -> None:
        """create a CALLS relationship between functions."""
        query = """
        MATCH (caller:Function {qualified_name: $caller_qualified})
        MERGE (callee:Function {name: $callee_name})
        ON CREATE SET callee.inferred = true
        MERGE (caller)-[r:CALLS]->(callee)
        SET r.line_number = $line_number,
            r.call_expression = $call_expression,
            r.confidence = $confidence,
            r.is_cross_file = $is_cross_file
        """
        graph.query(
            query,
            params={
                "caller_qualified": edge.caller_qualified,
                "callee_name": edge.callee_name,
                "line_number": edge.line_number,
                "call_expression": edge.call_expression,
                "confidence": edge.confidence,
                "is_cross_file": edge.is_cross_file,
            },
        )

    def _upsert_inheritance_edge(self, graph: "Graph", edge: InheritanceEdge) -> None:
        """create an INHERITS relationship between classes."""
        rel_type = "IMPLEMENTS" if edge.inheritance_type == "implements" else "INHERITS"

        query = f"""
        MATCH (child:Class {{qualified_name: $child_qualified}})
        MERGE (parent:Class {{name: $parent_name}})
        ON CREATE SET parent.inferred = true
        MERGE (child)-[r:{rel_type}]->(parent)
        SET r.is_cross_file = $is_cross_file
        """

        graph.query(
            query,
            params={
                "child_qualified": edge.child_qualified,
                "parent_name": edge.parent_name,
                "is_cross_file": edge.is_cross_file,
            },
        )

    def _upsert_import_edge(self, graph: "Graph", imp: ResolvedImport) -> None:
        """create an IMPORTS relationship between files/modules."""
        query = """
        MATCH (source:File {file_id: $source_file_id})
        MERGE (target:File {file_id: $target_file_id})
        MERGE (source)-[r:IMPORTS]->(target)
        SET r.names = $names,
            r.line_number = $line_number
        """
        graph.query(
            query,
            params={
                "source_file_id": imp.source_file_id,
                "target_file_id": imp.target_file_id,
                "names": imp.imported_names,
                "line_number": imp.line_number,
            },
        )

    def create_indexes(self) -> None:
        """create indexes for the graph."""
        graph = self._get_graph()

        graph.create_node_range_index("File", "file_id")
        graph.create_node_range_index("File", "path")
        graph.create_node_range_index("Function", "qualified_name")
        graph.create_node_range_index("Function", "name")
        graph.create_node_range_index("Class", "qualified_name")
        graph.create_node_range_index("Class", "name")

        graph.create_node_vector_index(
            "Function",
            "embedding",
            dim=768,
            similarity_function="cosine",
        )
        graph.create_node_vector_index(
            "Class",
            "embedding",
            dim=768,
            similarity_function="cosine",
        )

        logger.info(f"Created indexes for graph '{self._graph_name}'")

    async def semantic_search(
        self,
        query_embedding: list[float],
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> GraphQueryResult:
        """
        search for nodes by embedding similarity.
        """
        graph = self._get_graph()

        labels = []
        if node_types:
            labels = [nt.value for nt in node_types]
        else:
            labels = ["Function", "Class"]

        results = []
        for label in labels:
            query = f"""
            MATCH (n:{label})
            WHERE n.file_id IN $file_ids OR $file_ids IS NULL
            CALL vector_search(n.embedding, $query_embedding, $limit)
            YIELD node, score
            RETURN node, score
            ORDER BY score ASC
            LIMIT $limit
            """
            params = {
                "query_embedding": query_embedding,
                "file_ids": file_ids,
                "limit": limit,
            }
            result = graph.query(query, params=params)

            for row in result.result_set:
                node_data = row[0] if row else None
                score = row[1] if len(row) > 1 else 0.0

                if node_data:
                    code_node = self._node_to_code_node(node_data)
                    results.append((code_node, score))

        results.sort(key=lambda x: x[1])
        top_results = results[:limit]

        return GraphQueryResult(
            nodes=[r[0] for r in top_results],
            total_count=len(results),
            query_type="semantic",
        )

    def structural_query(
        self,
        query_type: str,
        name: str | None = None,
        file_id: int | None = None,
        depth: int = 3,
        limit: int = 50,
    ) -> GraphQueryResult:
        """
        execute structural queries on the graph.
        query_type: 'callers', 'callees', 'call_chain', 'class_symbols', 'inheritance'
        """
        graph = self._get_graph()
        nodes = []
        paths = []

        if query_type == "callers" and name:
            query = """
            MATCH (caller:Function)-[:CALLS]->(f:Function)
            WHERE f.name = $name OR f.qualified_name = $name
            RETURN caller
            LIMIT $limit
            """
            result = graph.query(query, params={"name": name, "limit": limit})
            for row in result.result_set:
                nodes.append(self._node_to_code_node(row[0]))

        elif query_type == "callees" and name:
            query = """
            MATCH (f:Function {name: $name})-[:CALLS]->(callee:Function)
            RETURN callee
            LIMIT $limit
            """
            result = graph.query(query, params={"name": name, "limit": limit})
            for row in result.result_set:
                nodes.append(self._node_to_code_node(row[0]))

        elif query_type == "call_chain" and name:
            query = f"""
            MATCH path = (f:Function {{name: $name}})-[:CALLS*1..{depth}]->(dep)
            RETURN path
            LIMIT $limit
            """
            result = graph.query(
                query, params={"name": name, "depth": depth, "limit": limit}
            )
            for row in result.result_set:
                path_nodes = []
                path = row[0]
                for i in range(path.node_count()):
                    node = path.get_node(i)
                    path_nodes.append(self._node_to_code_node(node))
                if path_nodes:
                    paths.append(path_nodes)

        elif query_type == "class_symbols" and file_id:
            query = """
            MATCH (file:File {file_id: $file_id})-[:CONTAINS]->(node)
            RETURN node
            LIMIT $limit
            """
            result = graph.query(query, params={"file_id": file_id, "limit": limit})
            for row in result.result_set:
                nodes.append(self._node_to_code_node(row[0]))

        elif query_type == "inheritance" and name:
            query = f"""
            MATCH path = (c:Class {{name: $name}})-[:INHERITS|IMPLEMENTS*1..{depth}]->(base)
            RETURN path
            LIMIT $limit
            """
            result = graph.query(
                query, params={"name": name, "depth": depth, "limit": limit}
            )
            for row in result.result_set:
                path_nodes = []
                path = row[0]
                for i in range(path.node_count()):
                    node = path.get_node(i)
                    path_nodes.append(self._node_to_code_node(node))
                if path_nodes:
                    paths.append(path_nodes)

        return GraphQueryResult(
            nodes=nodes,
            paths=paths,
            total_count=len(nodes) + len(paths),
            query_type="structural",
        )

    def _node_to_code_node(self, node: Node) -> CodeNode:
        """convert a falkordb node to a code node."""
        props = node.properties if hasattr(node, "properties") else {}

        label = "Function"
        if hasattr(node, "labels") and node.labels:
            label = node.labels[0]

        node_type = GraphNodeType.FUNCTION
        if label == "Class":
            node_type = GraphNodeType.CLASS
        elif label == "Method":
            node_type = GraphNodeType.METHOD
        elif label == "File":
            node_type = GraphNodeType.FILE

        return CodeNode(
            node_type=node_type,
            name=props.get("name", ""),
            qualified_name=props.get("qualified_name", ""),
            file_id=props.get("file_id", 0),
            file_path=props.get("file_path", ""),
            line_start=props.get("line_start", 0),
            line_end=props.get("line_end", 0),
            docstring=props.get("docstring"),
            embedding=props.get("embedding"),
        )

    def clear_graph(self) -> None:
        """delete all nodes and edges in the graph."""
        graph = self._get_graph()
        graph.query("MATCH (n) DETACH DELETE n")
        logger.info(f"Cleared graph '{self._graph_name}'")
