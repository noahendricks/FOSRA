"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ulid
from falkordb.falkordb import FalkorDB

from backend.src.domain.schemas.graph import (
    CodeNode,
    FunctionMetadata,
)
from backend.src.domain.schemas.treesitter_types import (
    Node,
    Range,
)
from backend.src.services.processing.utils.flatten import (
    FlattenNodes,
    extract_call_edges,
    extract_inheritance,
    extract_method_edges,
)
from backend.src.services.processing.utils.parse_utils import (
    CALL_QUERY_PATTERNS,
    CLASS_QUERY_PATTERNS,
    LANGUAGE_MODULES,
    _module_name_from_source_path,
    get_language,
)
from tree_sitter import Language, Parser
from tree_sitter import Node as NodeTS
import tree_sitter_python
from backend.src.services.processing.utils.parsing_funcs import (
    parse_node,
)
from backend.src.services.retrieval.graph_service import GraphService
from backend.src.settings.config import EmbedderConfig
from tree_sitter import Query, QueryCursor


async def parse_file(source_file: Path, name: str) -> Any:
    """test parse_node / parse_compound_statement on a source file."""
    python_lang = Language(tree_sitter_python.language())

    parser = Parser(python_lang)

    file_id = str(ulid.ULID())

    code = source_file.read_text()
    tree = parser.parse(bytes(code, "utf-8"))
    root = tree.root_node

    module_name = _module_name_from_source_path(source_file)

    # compute file path relative to project root for consistent import resolution
    # project root is the parent of 'backend' (or cwd if already at project root)
    project_root = Path.cwd()
    if project_root.name == "backend":
        project_root = project_root.parent

    absolute_path = source_file.absolute()
    try:
        # Make path relative to project root
        file_path = str(absolute_path.relative_to(project_root))
    except ValueError:
        # Fallback to absolute path if not under project root
        file_path = absolute_path.as_posix()

    # root node — module level
    root_node = Node(
        identifier=module_name,
        path=module_name,
        parent_id="",
        type="module",
        range=Range.from_node(root),
        text="",
        children=[],
        comments=[],
        file_id=file_id,
    )

    def parse_tree(node: NodeTS, parent: Node, name: str) -> Node:
        """Recursively parse tree-sitter node into our Node structure."""
        node_type = node.type

        # skip the root module node itself — its children become our root's children
        if node_type == "module":
            for child in node.named_children:
                parsed = parse_node(child, parent=parent)
                if parsed is not None:
                    parent.children.append(parsed)
            return parent

        # for all other nodes, parse them
        parsed = parse_node(node, parent=parent)

        if parsed is None:
            return parent

        # if parsed is a compound node with body, parse its children into body.statements
        # and use parsed as the new parent for proper identifier scoping
        if parsed.type in (
            "class_definition",
            "function_definition",
            "decorated_definition",
        ):
            body = getattr(parsed, "body", None)
            if body is not None and body.statements:
                # parse children into body statements using parsed as parent for scoping
                if body.statements:
                    parsed.children = list(body.statements)

        return parent

    result = parse_tree(root, parent=root_node, name=name)

    graph = FlattenNodes(
        file_id=file_id,
        file_path=file_path,
        source_content=code,
        ts_root=root,
    ).flatten_root(result, path=file_path, ts_root=root)

    falkor = FalkorDB()
    graph_service = GraphService(client=falkor)

    upsert = await graph_service.upsert_file_graph(
        graph_result=graph, embedder_config=EmbedderConfig()
    )
    return upsert
