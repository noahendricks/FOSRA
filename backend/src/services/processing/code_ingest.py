"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tree_sitter_python
import ulid
from falkordb.falkordb import FalkorDB
from tree_sitter import Language, Parser
from tree_sitter import Node as NodeTS

from backend.src.domain.schemas.graph import GraphResult
from backend.src.domain.schemas.graph_types import ImportNode
from backend.src.domain.schemas.treesitter_types import Node, Range
from backend.src.services.processing.utils.graph_extraction import (
    extract_call_edges,
    extract_inheritance,
    extract_method_edges,
)
from backend.src.services.processing.utils.parse_utils import (
    _module_name_from_source_path,
)
from backend.src.services.processing.utils.parsing_funcs import (
    _to_code_node,
    parse_node,
)
from backend.src.services.retrieval.graph_service import GraphService
from backend.src.settings.config import EmbedderConfig


def _collect_code_nodes(root: Node, file_path: str) -> list[Any]:
    """collect all CodeNodes from the parsed tree."""
    code_nodes: list[Any] = []

    def walk(node: Any) -> None:
        if not isinstance(node, Node):
            return
        code_node = _to_code_node(node, file_path)
        if code_node is not None:
            code_nodes.append(code_node)
        for child in getattr(node, "children", []):
            walk(child)

    walk(root)
    return code_nodes


def _collect_imports(root: Node) -> list[ImportNode]:
    """collect all ImportNodes from the parsed tree."""
    imports: list[ImportNode] = []

    def walk(node: Any) -> None:
        if isinstance(node, ImportNode):
            imports.append(node)
        elif isinstance(node, Node):
            for child in getattr(node, "children", []):
                walk(child)

    walk(root)
    return imports


def extract_graph(
    source_code: str,
    file_path: str,
    file_id: str,
    language: str = "python",
) -> GraphResult:
    python_lang = Language(tree_sitter_python.language())
    parser = Parser(python_lang)

    tree = parser.parse(bytes(source_code, "utf-8"))
    root = tree.root_node

    module_name = _module_name_from_source_path(Path(file_path))

    project_root = Path.cwd()
    if project_root.name == "backend":
        project_root = project_root.parent

    try:
        absolute_path = Path(file_path).absolute()
        if Path(file_path).is_absolute():
            file_path = str(absolute_path.relative_to(project_root))
    except ValueError:
        pass  # keep original file_path if relative fails

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

    def parse_tree(node: NodeTS, parent: Node) -> Node:
        node_type = node.type

        if node_type == "module":
            for child in node.named_children:
                parsed = parse_node(child, parent=parent)
                if parsed is not None:
                    parent.children.append(parsed)
            return parent

        parsed = parse_node(node, parent=parent)
        if parsed is None:
            return parent

        if parsed.type in (
            "class_definition",
            "function_definition",
            "decorated_definition",
        ):
            body = getattr(parsed, "body", None)
            if body is not None and body.statements:
                if body.statements:
                    parsed.children = list(body.statements)
        return parent

    result = parse_tree(root, parent=root_node)

    code_nodes = _collect_code_nodes(result, file_path)
    imports = _collect_imports(result)

    call_edges = extract_call_edges(
        root=root,
        file_id=file_id,
        source_code=source_code,
        language=language,
        nodes=code_nodes,
    )
    method_edges = extract_method_edges(nodes=code_nodes, file_id=file_id)

    inheritance_edges = extract_inheritance(
        root=root,
        source_code=source_code,
        file_id=file_id,
        language=language,
        nodes=code_nodes,
    )

    return GraphResult(
        file_id=file_id,
        file_path=file_path,
        language=language,
        nodes=code_nodes,
        call_edges=call_edges,
        inheritance_edges=inheritance_edges,
        method_edges=method_edges,
        imports=imports,
    )


async def parse_file(source_file: Path, name: str) -> Any:
    """parse a source file and upsert its graph to falkordb."""
    file_id = str(ulid.ULID())
    code = source_file.read_text()

    project_root = Path.cwd()
    if project_root.name == "backend":
        project_root = project_root.parent

    absolute_path = source_file.absolute()
    try:
        file_path = str(absolute_path.relative_to(project_root))
    except ValueError:
        file_path = absolute_path.as_posix()
    graph = extract_graph(
        source_code=code,
        file_path=file_path,
        file_id=file_id,
        language="python",
    )

    falkor = FalkorDB()
    graph_service = GraphService(client=falkor)

    upsert = await graph_service.upsert_file_graph(
        graph_result=graph, embedder_config=EmbedderConfig()
    )
    return upsert
