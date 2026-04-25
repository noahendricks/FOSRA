from __future__ import annotations

from typing import TYPE_CHECKING, Any

from falkordb import FalkorDB, Node
from loguru import logger

from backend.src.domain.schemas.treesitter_types import (
    GraphNodeType,
    ImportNode,
)

if TYPE_CHECKING:
    from falkordb import Graph

    from backend.src.settings import EmbedderConfig

from backend.src.domain.schemas.doc import Chunk, ChunkMetadata
from backend.src.domain.schemas.graph import (
    CallEdge,
    ClassMetadata,
    CodeNode,
    FunctionMetadata,
    GraphQueryResult,
    GraphResult,
    ImportMetadata,
    InheritanceEdge,
    MethodEdge,
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

    def create_indexes(self) -> None:
        graph = self._get_graph()
        graph.query("CALL db.idx.fulltext.createNodeIndex('File', 'file_id')")
        logger.info("Created full-text index on File.file_id")

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

        # Create or update the File node first
        self._upsert_file_node(graph, graph_result)
        stats["nodes_created"] += 1

        for node in graph_result.nodes:
            self._upsert_node(graph, node, node_embeddings.get(node.qualified_name))
            self._upsert_contains_edge(graph, node)
            stats["nodes_created"] += 1
            stats["edges_created"] += 1  # CONTAINS edge for each node

        for edge in graph_result.call_edges:
            self.upsert_call_edge(graph, edge)
            stats["edges_created"] += 1

        for edge in graph_result.method_edges:
            self.upsert_method_edge(graph, edge)
            stats["edges_created"] += 1

        for edge in graph_result.inheritance_edges:
            self.upsert_inheritance_edge(graph, edge)
            stats["edges_created"] += 1

        # extract imports (after full upsers)
        resolved_stats = await self.resolve_imports(
            imports=[n for n in graph_result.imports if not n.target_file_id]
        )

        for resolved_imp in graph_result.imports:
            if resolved_imp.target_file_id:
                self._upsert_import_edge(graph, resolved_imp)
                stats["edges_created"] += 1

        logger.bind(_structured={"file_path": graph_result.file_path, **stats}).info(
            "Upserted file"
        )
        return stats

    def _upsert_import_edge(self, graph: "Graph", imp: ImportNode) -> None:
        query = """
        MATCH (source:File {file_id: $source_file_id})
        MATCH (target:File {file_id: $target_file_id})
        MERGE (source)-[r:IMPORTS]->(target)
        SET r.names = $names,
            r.line_number = $line_number
        """
        _ = graph.query(
            query,
            params={
                "source_file_id": imp.file_id,
                "target_file_id": imp.target_file_id,
                "names": imp.import_dotted_names,
                "line_number": imp.line_number,
            },
        )

    def _upsert_uses_edges(
        self,
        graph: "Graph",
        source_file_id: str,
        target_file_id: str,
        imported_names: list[str],
        line_number: int | None,
    ) -> int:
        """create USES edges from File to items in target file."""
        if not imported_names or not target_file_id:
            return 0

        edges_created = 0

        for name in imported_names:
            # look for this name in the target file (Class, Function, Constant)
            for label in ["Class", "Function", "Constant"]:
                query = f"""
                MATCH (source:File {{file_id: $source_file_id}})
                MATCH (target:{label} {{file_id: $target_file_id, name: $name}})
                MERGE (source)-[r:USES]->(target)
                SET r.imported_names = $imported_names,
                    r.line_number = $line_number
                """
                result = graph.query(
                    query,
                    params={
                        "source_file_id": source_file_id,
                        "target_file_id": target_file_id,
                        "name": name,
                        "imported_names": imported_names,
                        "line_number": line_number or 0,
                    },
                )
                if result.result_set:
                    edges_created += 1

        return edges_created

    def upsert_method_edge(self, graph: "Graph", edge: MethodEdge) -> None:
        """create a DEFINES_METHOD relationship between a class and its method."""
        query = """
        MATCH (cls:Class {qualified_name: $class_qualified})
        MERGE (method:Method {qualified_name: $method_qualified})
        ON CREATE SET method.name = $method_name,
            method.file_id = $method_file_id,
            method.inferred = true
        MERGE (cls)-[r:DEFINES_METHOD]->(method)
        """
        _ = graph.query(
            query,
            params={
                "class_qualified": edge.class_qualified,
                "method_name": edge.method_name,
                "method_qualified": edge.method_qualified or edge.method_name,
                "method_file_id": edge.method_file_id or "",
            },
        )

    def upsert_call_edge(self, graph: "Graph", edge: CallEdge) -> None:
        """Create a CALLS relationship between functions/methods."""
        # try to match both Function and Method nodes for caller/callee
        query = """
        MATCH (caller)
        WHERE (caller:Function OR caller:Method) AND caller.qualified_name = $caller_qualified
        MERGE (callee {qualified_name: $callee_qualified})
        ON CREATE SET callee.name = $callee_name,
            callee.file_id = $callee_file_id,
            callee.inferred = true
        MERGE (caller)-[r:CALLS]->(callee)
        SET r.line_number = $line_number,
            r.call_expression = $call_expression,
            r.confidence = $confidence,
            r.is_cross_file = $is_cross_file
        """
        # only create the edge if we have a callee_qualified (don't create for built-ins)
        if not edge.callee_qualified:
            return
        _ = graph.query(
            query,
            params={
                "caller_qualified": edge.caller_qualified,
                "callee_name": edge.callee_name,
                "callee_qualified": edge.callee_qualified or edge.callee_name,
                "callee_file_id": edge.callee_file_id or "",
                "line_number": edge.line_number,
                "call_expression": edge.call_expression,
                "confidence": edge.confidence,
                "is_cross_file": edge.is_cross_file,
            },
        )

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

        if isinstance(node.metadata, FunctionMetadata):
            containing_class = node.metadata.containing_class
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
                metadata=node.metadata,
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
                metadata=node.metadata,
            )
            props["signature_str"] = sig_node._signature_to_string()
            props["is_async"] = node.signature.is_async
            sig_dict = {
                "parameters": node.signature.parameters.to_dict()
                if node.signature.parameters
                else {},
                "return_type": node.signature.return_type,
                "is_async": node.signature.is_async,
                "is_method": node.signature.is_method,
                "receiver": node.signature.receiver,
                "decorators": node.signature.decorators,
            }
            props["signature"] = json.dumps(sig_dict)

        if node.source_code:
            props["source_code"] = node.source_code

        # store metadata for Import nodes
        if node.node_type == GraphNodeType.IMPORT and isinstance(
            node.metadata, ImportMetadata
        ):
            metadata_dict = {}
            if node.metadata.imported_names:
                metadata_dict["imported_names"] = node.metadata.imported_names
            if node.metadata.aliased:
                metadata_dict["aliased"] = node.metadata.aliased
            if node.metadata.is_relative:
                metadata_dict["is_relative"] = node.metadata.is_relative
            if node.metadata.is_wildcard:
                metadata_dict["is_wildcard"] = node.metadata.is_wildcard
            if metadata_dict:
                props["imported_names"] = node.metadata.imported_names

        if embedding:
            # store embedding using vecf32() for FalkorDB vector index
            import json

            emb_str = json.dumps(embedding)

            emb_query = f"""
            WITH $qualified_name as qn, $props as p, vecf32({emb_str}) as vec
            MERGE (n:{label} {{qualified_name: qn}})
            SET n += p,
                n.embedding = vec
            """
            _ = graph.query(
                emb_query,
                params={"qualified_name": node.qualified_name, "props": props},
            )
            return

    def _upsert_contains_edge(
        self,
        graph: "Graph",
        node: CodeNode,
    ) -> None:
        label = node.node_type.value
        query = f"""
        MATCH (file:File {{file_id: $file_id}})
        MERGE (n:{label} {{qualified_name: $qualified_name}})
        MERGE (file)-[r:CONTAINS]->(n)
        """
        _ = graph.query(
            query,
            params={
                "file_id": node.file_id,
                "qualified_name": node.qualified_name,
            },
        )

    def _upsert_file_node(
        self,
        graph: "Graph",
        graph_result: GraphResult,
    ) -> None:
        """create or update the File node for a graph result."""
        query = """
        MERGE (file:File {file_id: $file_id})
        SET file.name = $name,
            file.path = $path,
            file.language = $language
        """
        _ = graph.query(
            query,
            params={
                "file_id": graph_result.file_id,
                "name": graph_result.file_path.split("/")[-1],
                "path": graph_result.file_path,
                "language": graph_result.language,
            },
        )

    async def semantic_search(
        self,
        query_embedding: list[float],
        node_types: list[GraphNodeType] | None = None,
        file_ids: list[int] | None = None,
        limit: int = 20,
    ) -> GraphQueryResult:
        """
        search for nodes by embedding similarity using FalkorDB vector index.
        """
        graph = self._get_graph()
        import json

        if node_types:
            labels = [nt.value for nt in node_types]
        else:
            labels = ["Function", "Method", "Class"]

        results = []

        # Build vecf32 query vector
        emb_str = json.dumps(query_embedding)

        # Try using vector index first, fall back to brute force
        for label in labels:
            try:
                # Use FalkorDB's vector index for ANN search
                query = f"""
                CALL db.idx.vector.queryNodes($label, 'embedding', $k, vecf32({emb_str}))
                YIELD node, score
                RETURN node, score
                """
                result = graph.query(
                    query,
                    params={"label": label, "k": limit * 2},
                )

                for row in result.result_set:
                    if not row or len(row) < 2:
                        continue
                    node_data = row[0]
                    vector_score = row[1]  # Already similarity score from vector index

                    code_node = self._node_to_code_node(node_data)
                    if file_ids is not None and code_node.file_id not in {
                        str(fid) for fid in file_ids
                    }:
                        continue

                    results.append((code_node, vector_score))
            except Exception:
                # Fall back to brute force if vector index fails
                try:
                    import numpy as np

                    query = f"""
                    MATCH (n:{label})
                    WHERE n.embedding IS NOT NULL
                    RETURN n, n.embedding
                    LIMIT 200
                    """
                    result = graph.query(query)

                    q_emb = np.array(query_embedding, dtype=np.float32)
                    q_norm = np.linalg.norm(q_emb)
                    if q_norm == 0:
                        q_norm = 1.0

                    for row in result.result_set:
                        if not row or not row[0] or not row[1]:
                            continue
                        node_data = row[0]
                        emb_list = row[1]

                        code_node = self._node_to_code_node(node_data)
                        if file_ids is not None and code_node.file_id not in {
                            str(fid) for fid in file_ids
                        }:
                            continue

                        node_emb = np.array(emb_list, dtype=np.float32)
                        score = float(
                            np.dot(q_emb, node_emb)
                            / (q_norm * np.linalg.norm(node_emb) + 1e-8)
                        )
                        results.append((code_node, score))
                except Exception:
                    pass  # Skip this label if both fail

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
        """full-text keyword search on Function/Method/Class nodes."""
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
                    if file_ids is None or code_node.file_id in {
                        str(fid) for fid in file_ids
                    }:
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

        adaptive hop depth:
          N < 5  -> 2 hops
          5 <= N < 15 -> 1 hop
          N >= 15 -> no expansion
        follows CALLS, DEFINES_METHOD, INHERITS edges.
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
        """hybrid search: dense vector + full-text keyword, RRF fused."""
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
            MATCH (caller)-[:CALLS]->(f)
            WHERE (f:Function OR f:Method) AND (f.name = $name OR f.qualified_name = $name)
            RETURN caller
            LIMIT $limit
            """
            result = graph.query(query, params={"name": name, "limit": limit})
            for row in result.result_set:
                nodes.append(self._node_to_code_node(row[0]))

        elif query_type == "callees" and name:
            query = """
            MATCH (f)-[:CALLS]->(callee)
            WHERE (f:Function OR f:Method) AND f.name = $name
            RETURN callee
            LIMIT $limit
            """
            result = graph.query(query, params={"name": name, "limit": limit})
            for row in result.result_set:
                nodes.append(self._node_to_code_node(row[0]))

        elif query_type == "call_chain" and name:
            query = f"""
            MATCH path = (f)-[:CALLS*1..{depth}]->(dep)
            WHERE (f:Function OR f:Method) AND f.name = $name
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

        if label == "Method":
            node_type = GraphNodeType.METHOD
        elif label == "Function":
            node_type = GraphNodeType.FUNCTION
        elif label == "Class":
            node_type = GraphNodeType.CLASS
        elif label == "Import":
            node_type = GraphNodeType.IMPORT
        else:
            node_type = GraphNodeType.FUNCTION

        qualified_name = props.get("qualified_name", "")
        containing_class: str | None = None
        if node_type == GraphNodeType.METHOD:
            parts = qualified_name.rsplit(".", 1)
            if len(parts) == 2 and "." in parts[0]:
                containing_class = (
                    parts[0].rsplit(":", 1)[-1] if ":" in parts[0] else parts[0]
                )

        signature = None
        # Build metadata based on node type
        if node_type == GraphNodeType.METHOD:
            metadata = FunctionMetadata(containing_class=containing_class)
            sig_json = props.get("signature")
            if sig_json:
                try:
                    signature = Signature(**json.loads(props.get("signature", "{}")))
                except Exception:
                    signature = None
            else:
                signature = None
        elif node_type == GraphNodeType.FUNCTION:
            metadata = FunctionMetadata()
            sig_json = props.get("signature")
            if sig_json:
                try:
                    signature = Signature(**json.loads(props.get("signature", "{}")))
                except Exception:
                    signature = None
            else:
                signature = None
        elif node_type == GraphNodeType.CLASS:
            superclasses = []
            sig_json = props.get("signature")
            if sig_json:
                try:
                    sig_dict = json.loads(sig_json)
                    superclasses = sig_dict.get("superclasses", [])
                    signature = Signature(
                        **sig_json.loads(props.get("signature", "{}"))
                    )
                except Exception:
                    pass
            else:
                signature = None
            metadata = ClassMetadata(superclasses=superclasses)
        elif node_type == GraphNodeType.IMPORT:
            metadata = ImportMetadata()
        else:
            metadata = FunctionMetadata()

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

    async def resolve_imports(
        self,
        imports: list[ImportNode],
    ) -> dict[str, Any]:
        graph = self._get_graph()
        stats = {"resolved": 0, "edges_created": 0, "failed": 0}

        unresolved = [
            imp for imp in imports if not imp.target_file_id or not imp.target_file_path
        ]

        if not unresolved:
            return stats

        for imp in unresolved:
            module_parts = imp.from_dotted_names if imp.from_dotted_names else []
            if not module_parts:
                stats["failed"] += 1
                continue

            # build the module path string (e.g., 'backend/src/domain/schemas/graph')
            file_path_pattern = "/".join(module_parts)

            query = """
                MATCH (f:File)
                WHERE f.path CONTAINS $pattern
                RETURN f.file_id, f.path
                LIMIT 1
                """

            result = graph.query(query, params={"pattern": file_path_pattern})
            rows = result.result_set

            if rows and rows[0]:
                imp.target_file_id = str(rows[0][0])
                imp.target_file_path = str(rows[0][1])
                stats["resolved"] += 1
                self._upsert_import_edge(graph, imp)
                stats["edges_created"] += 1

                # Create USES edges to specific imported items
                uses_edges = self._upsert_uses_edges(
                    graph,
                    source_file_id=imp.file_id,
                    target_file_id=imp.target_file_id,
                    imported_names=imp.import_dotted_names or [],
                    line_number=imp.line_number,
                )
                stats["edges_created"] += uses_edges
            else:
                stats["failed"] += 1

        logger.info(
            "Resolved {} imports ({} edges created, {} failed)",
            stats["resolved"],
            stats["edges_created"],
            stats["failed"],
        )
        return stats


    def upsert_inheritance_edge(self, graph: "Graph", edge: InheritanceEdge) -> None:
        """Create an EXTENDS relationship between classes."""
        query = """
        MATCH (child)
        WHERE child:Class AND child.qualified_name = $child_qualified
        MERGE (parent {qualified_name: $parent_qualified})
        ON CREATE SET parent.name = $parent_name,
            parent.file_id = $parent_file_id,
            parent.inferred = true
        MERGE (child)-[r:EXTENDS]->(parent)
        SET r.inheritance_type = $inheritance_type
        """
        # only create the edge if we have a child_qualified
        if not edge.child_qualified:
            return
        _ = graph.query(
            query,
            params={
                "child_qualified": edge.child_qualified,
                "child_name": edge.child_name,
                "parent_name": edge.parent_name,
                "parent_qualified": edge.parent_qualified or edge.parent_name,
                "parent_file_id": edge.parent_file_id or "",
                "inheritance_type": edge.inheritance_type,
            },
        )
