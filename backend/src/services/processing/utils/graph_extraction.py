"""tree-sitter query-based graph edge extraction.

extracts call edges, inheritance edges, and method edges using tree-sitter queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tree_sitter import Query, QueryCursor

from backend.src.domain.schemas.graph import CallEdge, CodeNode, FunctionMetadata, InheritanceEdge, MethodEdge
from backend.src.domain.schemas.graph_types import (
    GraphNodeType,
    ImportNode,
)
from backend.src.services.processing.utils.parse_utils import (
    CALL_QUERY_PATTERNS,
    CLASS_QUERY_PATTERNS,
    get_language,
)

if TYPE_CHECKING:
    import tree_sitter as ts

    from backend.src.domain.schemas.graph import CallEdge, CodeNode, InheritanceEdge, MethodEdge


# Exception/base class names for is_exception detection
EXCEPTION_CLASSES: frozenset[str] = frozenset(
    {
        "Exception",
        "BaseException",
        "Error",
        "Warning",
    }
)

# Abstract/base class decorators
ABSTRACT_DECORATORS: frozenset[str] = frozenset(
    {
        "abc.abstractmethod",
        "abstractmethod",
        "ABC",
    }
)

# Property decorators
PROPERTY_DECORATORS: frozenset[str] = frozenset(
    {
        "property",
        "cached_property",
        "functools.cached_property",
    }
)

# Override decorators
OVERRIDE_DECORATORS: frozenset[str] = frozenset(
    {
        "override",
        "typing.override",
        "overload",
    }
)

# Deprecated decorators
DEPRECATED_DECORATORS: frozenset[str] = frozenset(
    {
        "deprecated",
        "warnings.deprecated",
    }
)


def extract_call_edges(
    root: Any,
    source_code: str,
    file_id: str,
    language: str,
    nodes: list[CodeNode],
    resolved_imports: list[ImportNode] | None = None,
) -> list[CallEdge]:
    """extract call edges from tree-sitter queries."""
    from backend.src.domain.schemas.graph import CallEdge

    edges = []
    lang = get_language(language)

    query = Query(lang, CALL_QUERY_PATTERNS.get(language, ""))

    func_map = {}
    for node in nodes:
        if node.node_type in (GraphNodeType.FUNCTION, GraphNodeType.METHOD):
            func_map[node.line_start] = node

    import_lookup: dict[str, tuple[str, str | None]] = {}
    if resolved_imports:
        for imp in resolved_imports:
            for name in imp.import_dotted_names:
                import_lookup[name] = (
                    imp.target_file_path or "",
                    imp.target_file_id,
                )

    local_symbols: dict[str, str] = {}
    for node in nodes:
        if node.node_type in (GraphNodeType.FUNCTION, GraphNodeType.METHOD):
            local_symbols[node.name] = node.qualified_name

    cursor = QueryCursor(query)

    for pattern_idx, captures in cursor.matches(root):
        if "callee_name" in captures:
            for node in captures["callee_name"]:
                call_expr = node.parent
                if not call_expr:
                    continue

                callee_name = source_code[node.start_byte : node.end_byte]
                line_number = node.start_point[0] + 1

                caller_node = None

                for start_line, func in func_map.items():
                    if start_line <= line_number <= func.line_end:
                        caller_node = func
                        break

                if caller_node:
                    call_expression = source_code[
                        call_expr.start_byte : call_expr.end_byte
                    ]

                    callee_qualified: str | None = None
                    callee_file_id: str | None = None
                    is_cross_file = False

                    base_name = (
                        callee_name.split(".")[0] if "." in callee_name else callee_name
                    )

                    if base_name in import_lookup:
                        target_path, target_id = import_lookup[base_name]
                        if target_path:
                            if "." in callee_name:
                                method_name = callee_name.split(".")[-1]
                                callee_qualified = f"{target_path}:{method_name}"
                            else:
                                callee_qualified = f"{target_path}:{callee_name}"
                        if target_id:
                            callee_file_id = target_id
                            is_cross_file = caller_node.file_id != callee_file_id
                    elif callee_name in local_symbols:
                        callee_qualified = local_symbols[callee_name]
                        is_cross_file = False
                    elif base_name in ("self", "cls"):
                        method_name = (
                            callee_name.split(".", 1)[-1]
                            if "." in callee_name
                            else callee_name
                        )
                        if method_name in local_symbols:
                            callee_qualified = local_symbols[method_name]
                            is_cross_file = False

                    edges.append(
                        CallEdge(
                            caller_name=caller_node.name,
                            caller_qualified=caller_node.qualified_name,
                            caller_file_id=file_id,
                            callee_name=callee_name,
                            callee_qualified=callee_qualified,
                            callee_file_id=callee_file_id,
                            call_expression=call_expression,
                            line_number=line_number,
                            confidence=0.7,
                            is_cross_file=is_cross_file,
                        )
                    )

    return edges


def extract_inheritance(
    root: Any,
    source_code: str,
    file_id: str,
    language: str,
    nodes: list[CodeNode],
) -> list[InheritanceEdge]:
    """extract inheritance edges from tree-sitter queries."""
    from backend.src.domain.schemas.graph import InheritanceEdge

    edges = []
    lang = get_language(language)

    try:
        query = Query(lang, CLASS_QUERY_PATTERNS.get(language, ""))
    except Exception as ex:
        import logging
        logging.getLogger(__name__).warning(f"Invalid query pattern for {language}: {ex}")
        return edges

    class_map = {n.name: n for n in nodes if n.node_type == GraphNodeType.CLASS}

    cursor = QueryCursor(query)
    for pattern_idx, captures in cursor.matches(root):
        for capture_name in ("base", "extends", "implements"):
            if capture_name in captures:
                for node in captures[capture_name]:
                    class_node = node.parent
                    while class_node and class_node.type not in (
                        "class_definition",
                        "class_declaration",
                    ):
                        class_node = class_node.parent

                    if not class_node:
                        continue

                    name_node = None
                    for child in class_node.children:
                        if child.type == "identifier":
                            name_node = child
                            break

                    if not name_node:
                        continue

                    child_name = source_code[name_node.start_byte : name_node.end_byte]
                    child_qualified = class_map.get(child_name)
                    if not child_qualified:
                        child_qualified = f"{file_id}:{child_name}"
                    else:
                        child_qualified = child_qualified.qualified_name

                    parent_name = source_code[node.start_byte : node.end_byte]

                    edges.append(
                        InheritanceEdge(
                            child_name=child_name,
                            child_qualified=child_qualified,
                            child_file_id=file_id,
                            parent_name=parent_name,
                            inheritance_type=capture_name,
                        )
                    )

    return edges


def extract_method_edges(
    nodes: list[CodeNode],
    file_id: str,
) -> list[MethodEdge]:
    """extract method edges from a list of code nodes."""
    from backend.src.domain.schemas.graph import MethodEdge

    edges = []
    class_map: dict[str, list[CodeNode]] = {}

    for node in nodes:
        if node.node_type == GraphNodeType.CLASS:
            class_map[node.name] = [node]
        elif node.node_type == GraphNodeType.METHOD and isinstance(node.metadata, FunctionMetadata):
            containing_class = node.metadata.containing_class
            if containing_class:
                if containing_class not in class_map:
                    class_map[containing_class] = []
                class_map[containing_class].append(node)

    for class_name, members in class_map.items():
        if class_name not in [n.name for n in members if n.node_type == GraphNodeType.CLASS]:
            continue

        class_node = next(n for n in members if n.node_type == GraphNodeType.CLASS)

        for member in members:
            if member.node_type == GraphNodeType.METHOD:
                edges.append(
                    MethodEdge(
                        class_name=class_name,
                        class_qualified=class_node.qualified_name,
                        class_file_id=file_id,
                        method_name=member.name,
                        method_qualified=member.qualified_name,
                        method_file_id=file_id,
                    )
                )

    return edges
