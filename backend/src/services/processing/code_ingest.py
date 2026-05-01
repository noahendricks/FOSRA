"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import tree_sitter_python
import ulid
from falkordb.falkordb import FalkorDB
from tree_sitter import Language, Parser
from tree_sitter import Node as NodeTS

from backend.src.domain.schemas.doc import SectionMetadata, Subsection
from backend.src.domain.schemas.graph import GraphResult
from backend.src.domain.schemas.graph_types import (
    ClassMetadata,
    CodeNode,
    FunctionMetadata,
    GraphNodeType,
    ImportNode,
)
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


def _code_node_to_subsection(
    node: CodeNode,
    doc_id: str,
    doc_title: str,
) -> Subsection:
    """Convert a CodeNode to a Subsection for vector storage."""
    # Build text content from signature, docstring, and source
    content_parts = []

    # Add signature
    if node.signature:
        sig_str = node._signature_to_string()
        if sig_str:
            content_parts.append(sig_str)

    # Add docstring
    if node.docstring:
        content_parts.append(f'"""\n{node.docstring}\n"""')

    # Add source code snippet (first 30 lines)
    if node.source_code:
        lines = node.source_code.split("\n")
        content_parts.append("\n".join(lines[:30]))

    text = "\n\n".join(content_parts) if content_parts else node.name

    # Extract code-specific metadata
    is_async = False
    is_method = False
    decorators = None
    docstring = None
    parameters = None
    return_type = None
    code_def_type = None

    if isinstance(node.metadata, FunctionMetadata):
        is_async = node.metadata.is_async or False
        is_method = node.metadata.is_method or False
        decorators = node.metadata.decorators
        parameters = (
            [p.name for p in node.metadata.parameters.params]
            if node.metadata.parameters
            else None
        )
        return_type = node.metadata.return_type
        code_def_type = "function" if not is_method else "method"
    elif isinstance(node.metadata, ClassMetadata):
        decorators = node.metadata.decorators
        docstring = node.metadata.docstring
        code_def_type = "class"

    section_id = f"{doc_id}:{node.node_type.value}:{node.name}:{node.line_start}"

    metadata = SectionMetadata(
        section_id=section_id,
        doc_id=doc_id,
        doc_title=doc_title,
        token_count=len(text) // 4,
        source_file=node.file_path,
        code_definition_type=code_def_type,
        is_async=is_async,
        is_method=is_method,
        decorators=decorators,
        docstring=docstring or node.docstring,
        parameters=parameters,
        return_type=return_type,
    )

    return Subsection(text=text, metadata=metadata)


async def extract_code_chunks(
    source_code: str,
    file_path: str,
    language: str,
    doc_id: str | None = None,
    doc_title: str | None = None,
) -> list[Subsection]:
    """Extract code chunks from source code for vector storage.

    Parses source code with tree-sitter and converts CodeNodes to
    Subsections suitable for embedding and vector DB upsert.

    Args:
        source_code: The source code string to parse.
        file_path: Path identifier for the file.
        language: Programming language (python, javascript, typescript, go, rust).
        doc_id: Optional document ID (defaults to ULID).
        doc_title: Optional document title (defaults to file_path).

    Returns:
        List of Subsections, one per significant code node (class, function, method).
    """
    if doc_id is None:
        doc_id = str(uuid4())
    if doc_title is None:
        doc_title = file_path

    graph_result = extract_graph(
        source_code=source_code,
        file_path=file_path,
        file_id=doc_id,
        language=language,
    )

    chunks: list[Subsection] = []
    for node in graph_result.nodes:
        # Only create chunks for significant definition nodes
        if node.node_type in (
            GraphNodeType.CLASS,
            GraphNodeType.FUNCTION,
            GraphNodeType.METHOD,
        ):
            chunk = _code_node_to_subsection(node, doc_id, doc_title)
            chunks.append(chunk)

    return chunks


async def parse_file(
    source_file: Path,
) -> Any:
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
