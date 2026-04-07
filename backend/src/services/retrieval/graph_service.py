from __future__ import annotations

from typing import TYPE_CHECKING, Any

from falkordb import FalkorDB, Node
from loguru import logger

if TYPE_CHECKING:
    from falkordb import Graph

    from backend.src.settings import EmbedderConfig

from backend.src.domain.enums import GraphNodeType
from backend.src.domain.schemas.doc import Chunk, ChunkMetadata
from backend.src.domain.schemas.graph import (
    CallEdge,
    CodeNode,
    GraphQueryResult,
    GraphResult,
    InheritanceEdge,
    MethodEdge,
    ResolvedImport,
    Signature,
)
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

        for edge in graph_result.method_edges:
            self._upsert_method_edge(graph, edge)
            stats["edges_created"] += 1

        for imp in graph_result.imports:
            if imp.target_file_id:
                self._upsert_import_edge(graph, imp)
                stats["edges_created"] += 1

        logger.bind(_structured={"file_path": graph_result.file_path, **stats}).info(
            "Upserted file"
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
        """build the text to embed for a code node.

        For methods: prepend containing class context so the full method body
        is embedded with class semantics. Full body is embedded since Nomic
        handles 8192 tokens.
        """
        parts = []

        containing_class = node.metadata.get("containing_class")
        if containing_class:
            parts.append(f"class {containing_class}:")

        if node.signature:
            sig_node = CodeNode(
                node_type=node.node_type,
                name=node.name,
                qualified_name=node.qualified_name,
                file_id=node.file_id,
                file_path=node.file_path,
                line_start=node.line_start,
                line_end=node.line_end,
                signature=node.signature,
            )
            parts.append(sig_node._signature_to_string())

        if node.docstring:
            parts.append(node.docstring)

        if node.source_code:
            parts.append(node.source_code)

        return "\n\n".join(parts)

    def _upsert_node(
        self,
        graph: "Graph",
        node: CodeNode,
        embedding: list[float] | None,
    ) -> None:
        """create or update a node in the graph."""
        import json

        label = node.node_type.value

        props: dict[str, Any] = {
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
            sig_node = CodeNode(
                node_type=node.node_type,
                name=node.name,
                qualified_name=node.qualified_name,
                file_id=node.file_id,
                file_path=node.file_path,
                line_start=node.line_start,
                line_end=node.line_end,
                signature=node.signature,
            )
            props["signature_str"] = sig_node._signature_to_string()
            props["is_async"] = node.signature.is_async
            sig_dict = {
                "parameters": [
                    {
                        "name": p.name,
                        "type_annotation": p.type_annotation,
                        "default_value": p.default_value,
                        "is_variadic": p.is_variadic,
                        "is_keyword": p.is_keyword,
                    }
                    for p in node.signature.parameters
                ],
                "return_type": node.signature.return_type,
                "is_async": node.signature.is_async,
                "is_method": node.signature.is_method,
                "receiver": node.signature.receiver,
                "decorators": node.signature.decorators,
            }
            props["signature"] = json.dumps(sig_dict)

        if node.source_code:
            props["source_code"] = node.source_code

        if embedding:
            props["embedding"] = embedding

        query = f"""
        MERGE (n:{label} {{qualified_name: $qualified_name}})
        SET n += $props
        RETURN n
        """
        _ = graph.query(
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
        _ = graph.query(
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

        _ = graph.query(
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
        _ = graph.query(
            query,
            params={
                "source_file_id": imp.source_file_id,
                "target_file_id": imp.target_file_id,
                "names": imp.imported_names,
                "line_number": imp.line_number,
            },
        )

    def _upsert_method_edge(self, graph: "Graph", edge: MethodEdge) -> None:
        """create a DEFINES_METHOD relationship from Class to Method."""
        query = """
        MATCH (c:Class {qualified_name: $class_qualified})
        MATCH (m:Method {qualified_name: $method_qualified})
        MERGE (c)-[r:DEFINES_METHOD]->(m)
        """
        _ = graph.query(
            query,
            params={
                "class_qualified": edge.class_qualified,
                "method_qualified": edge.method_qualified,
            },
        )

    def create_indexes(self, embedder_config: "EmbedderConfig | None" = None) -> None:
        """create indexes for the graph (idempotent)."""
        graph = self._get_graph()

        for label, prop in [
            ("File", "file_id"),
            ("File", "path"),
            ("Function", "qualified_name"),
            ("Function", "name"),
            ("Class", "qualified_name"),
            ("Class", "name"),
        ]:
            try:
                _ = graph.create_node_range_index(label, prop)
            except Exception as e:
                if "already indexed" not in str(e).lower():
                    logger.warning(
                        "Index creation warning for {}.{}: {}", label, prop, e
                    )

        dim = embedder_config.dense_dimensions if embedder_config else 896
        for label, prop, vector_dim in [
            ("Function", "embedding", dim),
            ("Method", "embedding", dim),
            ("Class", "embedding", dim),
        ]:
            try:
                _ = graph.create_node_vector_index(
                    label,
                    prop,
                    dim=dim,
                    similarity_function="cosine",
                )
            except Exception as e:
                if "already indexed" not in str(e).lower():
                    logger.warning(
                        "Vector index creation warning for {}.{}: {}", label, prop, e
                    )

        for label in ("Function", "Method", "Class"):
            try:
                _ = graph.query(
                    f"CALL db.idx.fulltext.createNodeIndex('{label}', 'name', 'qualified_name', 'docstring')"
                )
            except Exception as e:
                if "already indexed" not in str(e).lower():
                    logger.warning(
                        "Fulltext index creation warning for {}: {}", label, e
                    )

        logger.info("Ensured indexes for graph '{}'", self._graph_name)

    async def semantic_search(
        self,
        query_embedding: list[float],
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> GraphQueryResult:
        """
        Search for nodes by embedding similarity.

        Since FalkorDB's db.idx.vector.queryNodes has compatibility issues with
        vecf32 parameter passing in this version, we fetch nodes with embeddings
        and compute cosine similarity in Python.
        """
        graph = self._get_graph()
        import numpy as np

        labels = []
        if node_types:
            labels = [nt.value for nt in node_types]
        else:
            labels = ["Function", "Method", "Class"]

        results = []
        q_emb = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            q_norm: float = 1.0

        label_placeholders = ",".join(f"'{lbl}'" for lbl in labels)
        query = f"""
        MATCH (n)
        WHERE labels(n)[0] IN [{label_placeholders}]
        AND n.embedding IS NOT NULL
        RETURN n, n.embedding
        LIMIT 200
        """
        result = graph.query(query)

        for row in result.result_set:
            if not row or not row[0]:
                continue
            node_data = row[0]
            emb_list = row[1]
            if not emb_list or len(emb_list) == 0:
                continue

            code_node = self._node_to_code_node(node_data)
            if file_ids is not None and int(code_node.file_id) not in file_ids:
                continue

            node_emb = np.array(emb_list, dtype=np.float32)
            score = float(
                np.dot(q_emb, node_emb) / (q_norm * np.linalg.norm(node_emb) + 1e-8)
            )
            results.append((code_node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:limit]

        return GraphQueryResult(
            nodes=[r[0] for r in top_results],
            total_count=len(results),
            query_type="semantic",
        )

    def _keyword_search(
        self,
        query: str,
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> list[tuple[CodeNode, float]]:
        """Full-text keyword search on Function/Method/Class nodes.

        Uses FalkorDB's db.idx.fulltext.queryNodes with RediSearch syntax.
        Returns nodes with TF-IDF scores (normalized to 0-1 by dividing by max).
        """
        graph = self._get_graph()

        labels = []
        if node_types:
            labels = [nt.value for nt in node_types]
        else:
            labels = ["Function", "Method", "Class"]

        terms = query.strip().split()
        if not terms:
            return []
        search_query = " ".join(terms)

        results: list[tuple[CodeNode, float]] = []
        max_score = 0.0

        for label in labels:
            cypher = f"""
            CALL db.idx.fulltext.queryNodes('{label}', $search_query)
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            LIMIT {limit}
            """
            result = graph.query(cypher, params={"search_query": search_query})

            for row in result.result_set:
                node_data = row[0] if row else None
                score = row[1] if len(row) > 1 else 0.0
                if node_data:
                    code_node = self._node_to_code_node(node_data)
                    if file_ids is None or int(code_node.file_id) in file_ids:
                        results.append((code_node, float(score)))
                        if abs(score) > max_score:
                            max_score = abs(score)

        if max_score > 0:
            results = [(n, abs(s) / max_score) for n, s in results]

        return results

    def _expand_via_graph(
        self,
        seed_nodes: list[CodeNode],
        file_ids: list[int] | None,
        limit: int = 20,
    ) -> list[CodeNode]:
        """Expand seed nodes via graph edges.

        Adaptive hop depth:
          N < 5  -> 2 hops
          5 <= N < 15 -> 1 hop
          N >= 15 -> no expansion
        Follows CALLS, DEFINES_METHOD, INHERITS edges.
        """
        if len(seed_nodes) >= 15:
            return []
        depth = 2 if len(seed_nodes) < 5 else 1

        graph = self._get_graph()
        seen: set[str] = {n.qualified_name for n in seed_nodes}
        frontier: list[str] = [n.qualified_name for n in seed_nodes]
        expanded: list[CodeNode] = []

        edge_types = ["CALLS", "DEFINES_METHOD", "INHERITS", "IMPLEMENTS"]
        edge_clause = "|".join(edge_types)

        for _ in range(depth):
            if not frontier:
                break
            if len(expanded) >= limit:
                break
            next_frontier: list[str] = []
            qn_list = frontier
            cypher = f"""
            MATCH (n) WHERE n.qualified_name IN $qns
            UNWIND [{",".join(f"'{et}'" for et in edge_types)}] AS etype
            MATCH (n)-[r]->(m) WHERE type(r) = etype
            RETURN DISTINCT m, n.qualified_name AS via
            LIMIT {limit}
            """
            params: dict[str, Any] = {"qns": qn_list}

            result = graph.query(cypher, params=params)

            for row in result.result_set:
                node_data = row[0] if row else None
                if node_data:
                    code_node = self._node_to_code_node(node_data)
                    qn = code_node.qualified_name
                    if qn not in seen:
                        if file_ids is None or code_node.file_id in {
                            str(fid) for fid in file_ids
                        }:
                            seen.add(qn)
                            expanded.append(code_node)
                        next_frontier.append(qn)

            frontier = next_frontier

        return expanded[:limit]

    async def hybrid_code_search(
        self,
        query: str,
        query_embedding: list[float],
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
        expand: bool = True,
    ) -> GraphQueryResult:
        """Hybrid search: dense vector + full-text keyword, RRF fused.

        Adaptive graph expansion is applied after initial results if expand=True.
        """
        RRF_K = 60
        dense_results = await self.semantic_search(
            query_embedding=query_embedding,
            node_types=node_types,
            file_ids=file_ids,
            limit=limit,
        )
        keyword_results = self._keyword_search(
            query=query,
            node_types=node_types,
            file_ids=file_ids,
            limit=limit,
        )

        chunk_scores: dict[str, tuple[CodeNode, float]] = {}
        for rank, node in enumerate(dense_results.nodes):
            key = node.qualified_name
            score = 1.0 / (RRF_K + rank)
            if key in chunk_scores:
                _, existing = chunk_scores[key]
                chunk_scores[key] = (node, existing + score)
            else:
                chunk_scores[key] = (node, score)

        for rank, (node, kw_score) in enumerate(keyword_results):
            key = node.qualified_name
            score = kw_score / (RRF_K + rank)
            if key in chunk_scores:
                _, existing = chunk_scores[key]
                chunk_scores[key] = (node, existing + score)
            else:
                chunk_scores[key] = (node, score)

        fused = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
        top_fused = fused[:limit]

        final_nodes = [n for n, _ in top_fused]

        if expand and final_nodes:
            expanded = self._expand_via_graph(final_nodes, file_ids, limit=limit)
            if expanded:
                expanded_qns = {n.qualified_name for n in final_nodes}
                for node in expanded:
                    if node.qualified_name not in expanded_qns:
                        final_nodes.append(node)
                        expanded_qns.add(node.qualified_name)
                final_nodes = final_nodes[:limit]

        return GraphQueryResult(
            nodes=final_nodes,
            total_count=len(final_nodes),
            query_type="hybrid",
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
        import json

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

        signature: Signature | None = None
        sig_json = props.get("signature")
        if sig_json:
            try:
                sig_dict = json.loads(sig_json)
                from backend.src.domain.schemas.graph import Parameter

                signature = Signature(
                    parameters=[
                        Parameter(
                            name=p["name"],
                            type_annotation=p.get("type_annotation"),
                            default_value=p.get("default_value"),
                            is_variadic=p.get("is_variadic", False),
                            is_keyword=p.get("is_keyword", False),
                        )
                        for p in sig_dict.get("parameters", [])
                    ],
                    return_type=sig_dict.get("return_type"),
                    is_async=sig_dict.get("is_async", False),
                    is_method=sig_dict.get("is_method", False),
                    receiver=sig_dict.get("receiver"),
                    decorators=sig_dict.get("decorators", []),
                )
            except Exception:
                pass

        qualified_name = props.get("qualified_name", "")
        containing_class: str | None = None
        if node_type == GraphNodeType.METHOD:
            parts = qualified_name.rsplit(".", 1)
            if len(parts) == 2 and "." in parts[0]:
                containing_class = (
                    parts[0].rsplit(":", 1)[-1] if ":" in parts[0] else parts[0]
                )
            metadata = {"containing_class": containing_class}
        else:
            metadata = {}

        return CodeNode(
            node_type=node_type,
            name=props.get("name", ""),
            qualified_name=qualified_name,
            file_id=str(props.get("file_id", "")),
            file_path=props.get("file_path", ""),
            line_start=props.get("line_start", 0),
            line_end=props.get("line_end", 0),
            docstring=props.get("docstring"),
            signature=signature,
            source_code=props.get("source_code"),
            embedding=props.get("embedding"),
            metadata=metadata,
        )

    def clear_graph(self) -> None:
        """delete all nodes and edges in the graph."""
        graph = self._get_graph()
        _ = graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared graph '{}'", self._graph_name)
