"""flatten treesitter AST nodes to graph-ready structures.

transformation layer: treesitter types -> flat graph nodes for storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tree_sitter import Query, QueryCursor

from backend.src.domain.schemas.graph import (
    CallEdge,
    ClassMetadata,
    CodeNode,
    ConstantMetadata,
    FunctionMetadata,
    GraphResult,
    ImportMetadata,
    InheritanceEdge,
    MethodEdge,
    Signature,
    TypeAliasMetadata,
)
from backend.src.domain.schemas.treesitter_types import (
    CLASS_TYPES,
    FUNCTION_TYPES,
    IMPORT_TYPES,
    SIMPLE,
    SIMPLE_TYPES,
    Block,
    ClassNode,
    Comment,
    DecoratedDefinition,
    FunctionNode,
    GraphNodeType,
    ImportNode,
    Node,
    Parameters,
    Point,
    Range,
    SimpleNode,
)
from backend.src.services.processing.utils.parse_utils import (
    CALL_QUERY_PATTERNS,
    CLASS_QUERY_PATTERNS,
    LANGUAGE_MODULES,
    get_language,
)
from backend.src.storage.utils.converters import DomainStruct

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


class FlattenNodes:
    """transform treesitter AST nodes to flat graph nodes for storage."""

    def __init__(
        self,
        file_id: str,
        file_path: str,
        language: str = "python",
        source_content: str | None = None,
        ts_root: Any = None,
    ):
        self.file_id = file_id
        self.file_path = file_path
        self.language = language
        self.source_content = source_content
        self.ts_root = ts_root

    def _extract_source_by_range(self, line_start: int, line_end: int) -> str | None:
        """extract source code by line range, preserving original indentation.

        tree-sitter's node.text strips leading whitespace from the first line of
        function definitions. this method uses line numbers to extract the exact
        original source including indentation.
        """
        if self.source_content is None:
            return None

        lines = self.source_content.split("\n")
        if line_start < 1 or line_end > len(lines):
            return None

        return "\n".join(lines[line_start - 1 : line_end])

    def flatten_body(self, body: Block) -> Node:
        """parses body block (list of statement nodes)
        into a single node with the text of all nodes
        and the info of its parent via the nodes"""
        if not body.statements:
            return Node(
                identifier="body",
                text="",
                path=self.file_path,
                range=Range(
                    start_point=Point(row=0, column=0),
                    end_point=Point(row=0, column=0),
                ),
                type="body",
                children=[],
            )

        # collect children and their text
        child_nodes: list[Node] = []
        parts: list[str] = []
        all_comments: list[Comment] = []

        for stmt in body.statements:
            if isinstance(stmt, Node):
                child_nodes.append(stmt)
                if stmt.text:
                    parts.append(stmt.text)
                if stmt.comments:
                    all_comments.extend(stmt.comments)

        # overall range spans from first child start to last child end
        first = child_nodes[0]
        last = child_nodes[-1]
        overall_range = Range(
            start_point=first.range.start_point,
            end_point=last.range.end_point,
            start_byte=first.range.start_byte,
            end_byte=last.range.end_byte,
        )

        # parent_id: use the last child's parent as the body's parent
        parent_id = last.parent_id

        body_text = "\n".join(parts)

        return Node(
            identifier="body",
            text=body_text.strip() if body_text else "",
            path=self.file_path,
            range=overall_range,
            type="body",
            parent_id=parent_id,
            children=child_nodes,
            comments=all_comments,
        )

    def flatten_function(
        self,
        node: FunctionNode | DecoratedDefinition,
    ) -> CodeNode | None:
        """flatten a function/method node to a graph-ready CodeNode."""
        func_node = self._unwrap_decorated(node)
        if not func_node:
            return None

        qualified_name = self._make_qualified(func_node)

        if isinstance(node, FunctionNode):
            params = node.parameters
            line_start = func_node.range.start_point.row + 1
            line_end = func_node.range.end_point.row + 1
            docstring = node.docstring.content if node.docstring else None
            source_code = (
                self._extract_source_by_range(line_start, line_end) or node.text
            )
            header = node.header
        elif isinstance(node, DecoratedDefinition) and node.definition:
            params = None
            line_start = node.definition.range.start_point.row + 1
            line_end = node.definition.range.end_point.row + 1
            docstring = (
                node.definition.docstring.content if node.definition.docstring else None
            )
            source_code = (
                self._extract_source_by_range(line_start, line_end)
                or node.definition.text
            )
            header = node.definition.header  # include decorators in header
        else:
            params = None
            source_code = ""
            docstring = None
            header = None

        signature = self._make_signature(
            params=params,
            return_type=func_node.return_type,
            is_async=func_node.is_async,
            receiver=func_node.receiver,
            decorators=func_node.decorators,
        )
        metadata = self._function_metadata(func_node)

        return CodeNode(
            node_type=GraphNodeType.METHOD
            if func_node.receiver
            else GraphNodeType.FUNCTION,
            name=func_node.name,
            qualified_name=qualified_name,
            file_id=self.file_id,
            file_path=self.file_path,
            line_start=func_node.range.start_point.row + 1,
            line_end=func_node.range.end_point.row + 1,
            signature=signature,
            docstring=docstring,
            source_code=source_code,
            metadata=metadata,
            header=header,
        )

    def flatten_class(self, node: ClassNode) -> CodeNode:
        """flatten a class node to a graph-ready CodeNode."""
        qualified_name = f"{self.file_path}:{node.name}"
        metadata = self._class_metadata(node)
        line_start = node.range.start_point.row + 1
        line_end = node.range.end_point.row + 1
        source_code = self._extract_source_by_range(line_start, line_end) or node.text

        return CodeNode(
            node_type=GraphNodeType.CLASS,
            name=node.name,
            qualified_name=qualified_name,
            file_id=self.file_id,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            docstring=node.docstring.content if node.docstring else None,
            source_code=source_code,
            metadata=metadata,
            header=node.header,
        )

    def flatten_import(self, node: ImportNode) -> CodeNode:
        """flatten an import node to a graph-ready CodeNode."""
        module_path = self._resolve_module_path(node)
        qualified_name = f"{self.file_path}:import:{module_path or 'unknown'}"
        metadata = self._import_metadata(node, module_path)

        return CodeNode(
            node_type=GraphNodeType.IMPORT,
            name=module_path or "unknown",
            qualified_name=qualified_name,
            file_id=self.file_id,
            file_path=self.file_path,
            line_start=node.range.start_point.row + 1,
            line_end=node.range.end_point.row + 1,
            source_code=node.text,
            metadata=metadata,
        )

    def flatten_constant(
        self,
        name: str,
        value_repr: str | None,
        value_type: str | None,
        line_start: int,
        line_end: int,
        source_code: str,
    ) -> CodeNode:
        """flatten a module-level constant/assignment to a graph-ready CodeNode."""
        qualified_name = f"{self.file_path}:{name}"

        return CodeNode(
            node_type=GraphNodeType.CONSTANT,
            name=name,
            qualified_name=qualified_name,
            file_id=self.file_id,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            source_code=source_code,
            metadata=ConstantMetadata(
                is_public=self._is_public(name, []),
                is_private=self._is_private(name),
                value_repr=value_repr,
                value_type=value_type,
            ),
        )

    class TypeAliasMetadata(DomainStruct):
        type_expr: str
        is_public: bool = False

    def flatten_type_alias(
        self,
        name: str,
        type_expr: str,
        line_start: int,
        line_end: int,
        source_code: str,
    ) -> CodeNode:
        """flatten a type alias to a graph-ready CodeNode."""
        qualified_name = f"{self.file_path}:{name}"
        return CodeNode(
            node_type=GraphNodeType.TYPE_ALIAS,
            name=name,
            qualified_name=qualified_name,
            file_id=self.file_id,
            file_path=self.file_path,
            line_start=line_start,
            line_end=line_end,
            source_code=source_code,
            metadata=TypeAliasMetadata(
                type_expr=type_expr, is_public=not name.startswith("_")
            ),
        )

    def flatten_node(self, node: Node) -> CodeNode | None:
        node_type = node.type

        # dispatch by node type
        if node_type in IMPORT_TYPES or isinstance(node, ImportNode):
            if isinstance(node, ImportNode):
                return self.flatten_import(node)

        elif (
            node_type in FUNCTION_TYPES
            or node_type == "decorated_definition"
            or isinstance(node, (FunctionNode, DecoratedDefinition))
        ):
            if isinstance(node, (FunctionNode, DecoratedDefinition)):
                return self.flatten_function(node)

        elif node_type in CLASS_TYPES or isinstance(node, ClassNode):
            if isinstance(node, ClassNode):
                return self.flatten_class(node)

        elif (
            (node_type in SIMPLE_TYPES or isinstance(node, SimpleNode))
            and isinstance(node, SimpleNode)
            and node.statement_type == SIMPLE.ASSIGNMENT
        ):
            # Extract name from identifier (format: module:name)
            name = node.identifier.split(":")[-1] if node.identifier else "unknown"
            return self.flatten_constant(
                name=name,
                value_repr=None,
                value_type=None,
                line_start=node.range.start_point.row + 1,
                line_end=node.range.end_point.row + 1,
                source_code=node.text or "",
            )

        return None

    def flatten_root(self, root: Node, path: str, ts_root: Any = None) -> GraphResult:
        """flatten the root node of a file to extract all top-level definitions.

        ts_root: optional tree-sitter root node for call edge extraction.
        if not provided, call edges will not be extracted.
        """
        code_nodes: list[CodeNode] = []

        # collect imports separately before flattening
        import_nodes: list[ImportNode] = []

        for child in root.children:
            if isinstance(child, Node):
                # Collect imports before flattening
                if isinstance(child, ImportNode):
                    import_nodes.append(child)

                # Flatten the node itself
                flattened = self.flatten_node(child)
                if flattened:
                    code_nodes.append(flattened)

                # Also flatten children inside classes (methods)
                if isinstance(child, ClassNode) and child.children:
                    for method in child.children:
                        if isinstance(method, Node):
                            method_flattened = self.flatten_node(method)
                            if method_flattened:
                                code_nodes.append(method_flattened)

        # extract call edges only if tree-sitter root is available
        call_edges: list[CallEdge] = []

        if ts_root is not None:
            call_edges = extract_call_edges(
                root=ts_root,
                file_id=self.file_id,
                source_code=self.source_content or "",
                language=self.language,
                nodes=code_nodes,
            )

        # extract method edges
        method_edges = extract_method_edges(nodes=code_nodes, file_id=self.file_id)

        graph_result = GraphResult(
            file_id=self.file_id,
            file_path=self.file_path,
            language=self.language,
            nodes=code_nodes,
            call_edges=call_edges,
            method_edges=method_edges,
            imports=import_nodes,
        )
        return graph_result

    # =========================================================================
    # Embedding text builder
    # =========================================================================

    def build_embedding_text(self, node: CodeNode) -> str:
        """build text for embedding that supports NL queries."""
        parts = []

        # PURPOSE: first line of docstring
        # PURPOSE: first line of docstring
        if node.docstring:
            first_line = node.docstring.split("\n")[0].strip()
            if first_line:
                parts.append(f"Purpose: {first_line}")

        # CONTEXT: module/class hierarchy
        context_parts = [f"file: {node.file_path}"]

        # Access metadata attributes directly based on type
        if (
            isinstance(node.metadata, FunctionMetadata)
            and node.metadata.containing_class
        ):
            context_parts.append(f"class: {node.metadata.containing_class}")
        if node.node_type == GraphNodeType.CLASS:
            if isinstance(node.metadata, ClassMetadata) and node.metadata.superclasses:
                context_parts.append(
                    f"inherits: {', '.join(node.metadata.superclasses)}"
                )
        if isinstance(node.metadata, ImportMetadata) and node.metadata.imported_names:
            context_parts.append(f"imports: {', '.join(node.metadata.imported_names)}")
        if context_parts:
            parts.append("Context: " + " | ".join(context_parts))

        # RELATIONSHIPS: decorators and characteristics
        rel_parts = []
        if isinstance(node.metadata, FunctionMetadata):
            for decorator in node.metadata.decorators:
                rel_parts.append(f"@{decorator}")
            if node.metadata.is_property:
                rel_parts.append("is a property")
            if node.metadata.is_override:
                rel_parts.append("overrides a parent method")
            if node.metadata.is_coroutine:
                rel_parts.append("is a coroutine")
            if node.metadata.is_abstract:
                rel_parts.append("is abstract")
        elif isinstance(node.metadata, ClassMetadata):
            for decorator in node.metadata.decorators:
                rel_parts.append(f"@{decorator}")
            if node.metadata.is_abstract:
                rel_parts.append("is abstract")
            if node.metadata.is_dataclass:
                rel_parts.append("is a dataclass")
            if node.metadata.is_enum:
                rel_parts.append("is an enum")
        if rel_parts:
            parts.append(f"Relationships: {', '.join(rel_parts)}")

        # DOCUMENTATION: full docstring
        if node.docstring:
            parts.append(f"Documentation: {node.docstring}")

        # SOURCE: full source for implementation details
        if node.source_code:
            parts.append(f"Implementation:\n{node.source_code}")

        return "\n\n".join(parts)

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _unwrap_decorated(
        self, node: FunctionNode | DecoratedDefinition
    ) -> FunctionNode | None:
        """unwrap DecoratedDefinition to FunctionNode."""
        if isinstance(node, DecoratedDefinition):
            return node.definition  # type:ignore
        if isinstance(node, FunctionNode):
            return node

    def _make_qualified(self, node: FunctionNode) -> str:
        """build qualified name for functions/methods.

        Format: <file_path>:<class_name>:<method_name> (with slashes in file path, colons as separators)
        """
        # node.identifier is like "src:services:processing:utils:flatten:py:module:FlattenNodes:my_method"
        parts = node.identifier.split(":")

        # Find the file end index - look for .py in segments or segment before 'py'
        file_end_idx = -1
        for i, part in enumerate(parts):
            if ".py" in part:
                file_end_idx = i
                break
        if file_end_idx < 0:
            # Check if 'py' is a separate segment (after the file name)
            for i in range(1, len(parts)):
                if parts[i] == "py" and "." not in parts[i - 1]:
                    file_end_idx = i
                    break

        if file_end_idx >= 0:
            # Build file path: parts[:file_end_idx+1], convert colons to slashes
            file_parts = parts[:file_end_idx]
            last_file_part = parts[file_end_idx]
            # Remove .py from last part if present
            if last_file_part.endswith(".py"):
                last_file_part = last_file_part[:-3]
            elif last_file_part == "py" and file_parts:
                # 'py' is a separate segment, use the previous file name part
                last_file_part = file_parts[-1]
                file_parts = file_parts[:-1]
            file_path = (
                "/".join(file_parts + [last_file_part])
                if file_parts
                else last_file_part
            )
            # Remaining parts are class/method names (filter out 'module')
            remaining = [p for p in parts[file_end_idx + 1 :] if p and p != "module"]
            if remaining:
                return f"{file_path}:{':'.join(remaining)}"
            return file_path

        # Fallback
        return node.identifier.replace(":module:", ":").replace(":py:", ":")

    def _make_signature(
        self,
        params: Parameters | None,
        return_type: str | None,
        is_async: bool,
        receiver: str | None,
        decorators: list[str],
    ) -> Signature:
        """build a signature from components."""
        return Signature(
            parameters=params if params else None,
            return_type=return_type,
            is_async=is_async,
            is_method=receiver is not None,
            receiver=receiver,
            decorators=decorators,
        )

    def _function_metadata(self, node: FunctionNode) -> FunctionMetadata:
        """extract metadata from a function node."""
        all_decorators = " ".join(node.decorators)

        return FunctionMetadata(
            containing_class=node.containing_class,
            is_public=self._is_public(node.name, node.decorators),
            is_private=self._is_private(node.name),
            is_property=any(d in PROPERTY_DECORATORS for d in node.decorators),
            is_coroutine=node.is_async,
            is_abstract=any(d in ABSTRACT_DECORATORS for d in node.decorators),
            is_override=any(d in OVERRIDE_DECORATORS for d in node.decorators),
            is_deprecated=any(d in DEPRECATED_DECORATORS for d in node.decorators),
            decorators=node.decorators,
            param_count=len(node.parameters.params) if node.parameters else 0,
            is_static="staticmethod" in all_decorators,
            is_overload="overload" in all_decorators,
        )

    def _class_metadata(self, node: ClassNode) -> ClassMetadata:
        """extract metadata from a class node."""
        all_decorators = " ".join(node.decorators)
        superclasses_str = " ".join(node.superclasses or [])

        return ClassMetadata(
            superclasses=node.superclasses or [],
            is_public=self._is_public(node.name, node.decorators),
            is_private=self._is_private(node.name),
            is_abstract=any(d in ABSTRACT_DECORATORS for d in node.decorators),
            is_exception=any(s in EXCEPTION_CLASSES for s in (node.superclasses or [])),
            decorators=node.decorators,
            method_count=len(node.methods),
            is_dataclass="dataclass" in all_decorators,
            is_enum="enum" in superclasses_str,
        )

    def _import_metadata(
        self, node: ImportNode, module_path: str | None
    ) -> ImportMetadata:
        """extract metadata from an import node."""
        is_relative = bool(module_path and module_path.startswith("."))

        return ImportMetadata(
            imported_names=node.import_dotted_names,
            aliased=node.aliased,
            is_relative=is_relative,
            is_wildcard="*" in node.import_dotted_names,
        )

    def _is_public(self, name: str, decorators: list[str]) -> bool:
        """determine visibility from name and decorators."""
        if name.startswith("__") and name.endswith("__"):
            return True  # dunder methods are "public"
        if name.startswith("_"):
            return False
        return True

    def _is_private(self, name: str) -> bool:
        """determine if name is private (single underscore prefix)."""
        return name.startswith("_") and not name.startswith("__")

    def _is_generator(self, body_text: str | None) -> bool:
        """check if body contains yield."""
        if not body_text:
            return False
        return "yield" in body_text

    def _resolve_module_path(self, node: ImportNode) -> str | None:
        """resolve the target module path from an import node."""
        if node.from_dotted_names:
            return ".".join(node.from_dotted_names)
        if node.import_dotted_names:
            return node.import_dotted_names[0]
        return None

    @staticmethod
    def get_module(file_path: str) -> str:
        """extract module path from file path.

        backend/src/services/graph.py -> backend.src.services.graph
        """
        parts = Path(file_path).parts
        if len(parts) >= 2:
            return ".".join(parts[:-1]) + "." + Path(file_path).stem
        return file_path

    @staticmethod
    def get_package(file_path: str) -> str:
        """extract top-level package from file path.

        backend/src/services/graph.py -> backend
        """
        parts = Path(file_path).parts
        return parts[0] if parts else ""

    @staticmethod
    def get_module_parts(file_path: str) -> list[str]:
        """extract module path parts from file path.

        backend/src/services/graph.py -> ['backend', 'src', 'services', 'graph']
        """
        path = Path(file_path)
        stem = path.stem
        if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            return list(path.parts[:-1]) + [stem]
        return list(path.parts)


# ── edge extraction helpers

# Moved from code_ingest.py to break the import cycle.
# FlatenNodes needs extract_call_edges / extract_method_edges but was
# lazily importing them back from code_ingest.py, creating a cycle.


def extract_call_edges(
    root: Any,
    source_code: str,
    file_id: str,
    language: str,
    nodes: list[CodeNode],
    resolved_imports: list[ImportNode] | None = None,
) -> list[CallEdge]:
    edges = []
    lang = get_language(language)

    query = Query(lang, CALL_QUERY_PATTERNS.get(language, ""))

    func_map = {}
    for node in nodes:
        if node.node_type in (GraphNodeType.FUNCTION, GraphNodeType.METHOD):
            func_map[node.line_start] = node

    # build import lookup: imported_name -> (target_file_path, target_file_id)
    import_lookup: dict[str, tuple[str, str | None]] = {}
    if resolved_imports:
        for imp in resolved_imports:
            for name in imp.import_dotted_names:
                import_lookup[name] = (
                    imp.target_file_path or "",
                    imp.target_file_id,
                )

    # build local symbol table: local_name -> qualified_name (for same-file calls)
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

                    # resolve callee: check imports first, then local symbols
                    callee_qualified: str | None = None
                    callee_file_id: str | None = None
                    is_cross_file = False

                    # handle method calls like `parser.parse()` - extract base name
                    base_name = (
                        callee_name.split(".")[0] if "." in callee_name else callee_name
                    )

                    if base_name in import_lookup:
                        target_path, target_id = import_lookup[base_name]
                        if target_path:
                            # build qualified name
                            if "." in callee_name:
                                # method call: use path:name for the method
                                method_name = callee_name.split(".")[-1]
                                callee_qualified = f"{target_path}:{method_name}"
                            else:
                                callee_qualified = f"{target_path}:{callee_name}"
                        if target_id:
                            callee_file_id = target_id
                            is_cross_file = caller_node.file_id != callee_file_id
                    elif callee_name in local_symbols:
                        # local call - same file
                        callee_qualified = local_symbols[callee_name]
                        is_cross_file = False
                    elif base_name in ("self", "cls"):
                        # method call like self.method() - extract method name
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
    edges = []
    lang = get_language(language=language)

    query = Query(lang, CLASS_QUERY_PATTERNS.get(language, ""))

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
                        if child.type in ("identifier", "type_identifier"):
                            name_node = child
                            break

                    if not name_node:
                        continue

                    child_name = source_code[name_node.start_byte : name_node.end_byte]
                    parent_name = source_code[node.start_byte : node.end_byte]

                    child_class = class_map.get(child_name)
                    if child_class:
                        edges.append(
                            InheritanceEdge(
                                child_name=child_name,
                                child_qualified=child_class.qualified_name,
                                child_file_id=file_id,
                                parent_name=parent_name,
                                parent_qualified=None,
                                parent_file_id=None,
                                inheritance_type="implements"
                                if capture_name == "implements"
                                else "extends",
                                is_cross_file=False,
                            )
                        )

    return edges


def extract_method_edges(
    nodes: list[CodeNode],
    file_id: str,
) -> list[MethodEdge]:
    """Create DEFINES_METHOD edges from Class nodes to their Method nodes."""
    edges = []
    class_map: dict[str, CodeNode] = {
        n.qualified_name: n for n in nodes if n.node_type == GraphNodeType.CLASS
    }

    for node in nodes:
        if (
            isinstance(node, CodeNode)
            and node.node_type == GraphNodeType.METHOD
            and isinstance(node.metadata, FunctionMetadata)
            and node.metadata.containing_class is not None
        ):
            class_name = node.metadata.containing_class
            class_node = class_map.get(class_name)
            if not class_node:
                # Try matching qualified_name ending with :class_name
                for qn, cn in class_map.items():
                    if qn.endswith(f":{class_name}"):
                        class_node = cn
                        break
            if class_node:
                edges.append(
                    MethodEdge(
                        class_name=class_name,
                        class_qualified=class_node.qualified_name,
                        class_file_id=file_id,
                        method_name=node.name,
                        method_qualified=node.qualified_name,
                        method_file_id=file_id,
                    )
                )
        elif isinstance(node, CodeNode) and isinstance(node.metadata, FunctionMetadata):
            # heuristic: if method name is found in class body, link it to the class
            for potential_class in class_map.values():
                if (
                    potential_class.file_id == file_id
                    and node.name in potential_class.header
                    if potential_class.header
                    else ""
                ):
                    edges.append(
                        MethodEdge(
                            class_name=potential_class.name,
                            class_qualified=potential_class.qualified_name,
                            class_file_id=file_id,
                            method_name=node.name,
                            method_qualified=node.qualified_name,
                            method_file_id=file_id,
                        )
                    )
                    break

    return edges
