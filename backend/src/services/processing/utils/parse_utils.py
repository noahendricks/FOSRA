from __future__ import annotations

import types
from pathlib import Path

import tree_sitter as ts
import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tsts
from tree_sitter import (
    Language,  # type: ignore
    Parser,  # type: ignore
)
from tree_sitter import Node as NodeTS

from backend.src.domain.schemas.graph_types import (
    Comment,
    Docstring,
    Node,
)

code_mimes = {
    "text/x-python": "python",
    "text/x-rust": "rust",
    "text/x-go": "go",
    "text/x-swift": "swift",
    "text/typescript": "typescript",
    "text/tsx": "tsx",
}

text_mimes = {
    "txt": "text/plain",
    "htm": "text/html",
    "html": "text/html",
    "css": "text/css",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "mdx": "text/markdown",
    "rtx": "text/richtext",
}


def _make_id(
    type_name: str,
    name: str | None,
    parent: Node,
) -> str:
    """build identifier in MODULE:OUTER_SCOPE:NAME format."""

    path = parent.path.rstrip(":.") if parent.path else ""

    if "." in path:
        path = path.replace(".", ":")

    base = path
    if base.endswith(":module"):
        base = base[:-7]  # remove ':module' (7 chars)
    elif base.endswith("module") and len(base) > 6:
        base = base[:-6]

    if not base:
        base = parent.identifier.rstrip(":")

    if name:
        result = f"{base}:{name}" if base else name
    else:
        type_part = type_name if "module" not in type_name.lower() else ""
        result = f"{base}:{type_part}" if type_part else base

    return result


def _get_body_block(node: NodeTS) -> NodeTS | None:
    """find the body block node — field name varies by statement type."""
    for field_name in ("body", "consequence"):
        block = node.child_by_field_name(field_name)
        if block is not None:
            return block
        return None


def _extract_comments(node: NodeTS) -> list[Comment]:
    """extract comment children from a tree-sitter node."""
    comments: list[Comment] = []
    for child in node.children:  # .children (not .named_children) includes comments
        if child.type == "comment":
            row = child.start_point.row
            # inline = same row as any named sibling
            is_inline = any(c.start_point.row == row for c in node.named_children)
            if child.text is not None:
                comments.append(
                    Comment(
                        text=child.text.decode("utf-8"),
                        row=row,
                        is_inline=is_inline,
                    )
                )
    return comments


def _extract_first_text(
    block: NodeTS | None,
    parent: Node,
    index: int = 0,
) -> tuple[Docstring | None, str] | None:
    """extract docstring from a block. returns (docstring, first_child_text_or_empty)."""
    if block is None:
        return None, ""

    first = block.children[0] if block.children else None

    if first is None:
        return None, ""

    if first.type != "expression_statement":
        return None, ""

    first_node = first.child_by_field_name("value") or (
        first.named_children[0] if first.named_children else None
    )
    if not first_node:
        return None

    # docstring exists, extract it
    if first_node and first_node.type == "string":
        # string_content is a named child, not a field
        string_content: str | None = None
        rest_content: str = ""
        for c in first_node.named_children:
            if c.type == "string_content":
                string_content = c.text.decode("utf-8") if c.text else None
            else:
                if c.text:
                    rest_content += c.text.decode("utf-8")

        if string_content is None:
            return None, ""

        # path represents ancestry: parent_path.body.index
        doc_path = f"{parent.path}.{index}" if parent.path else str(index)

        doc = Docstring(
            content=string_content,
            path=doc_path,
        )

    else:
        # doc is none, get all rest of node text
        rest_parts: list[str] = []

        for c in first_node.named_children:
            if c.type != "string_content" and c.text:
                rest_parts.append(c.text.decode("utf-8"))

        rest_content = "".join(rest_parts)

        doc = None

        if not rest_content:
            return None, ""

    return doc, first.text.decode("utf-8") if first.text else ""


def _node_type_to_name(node_type: str) -> str:
    """convert tree-sitter node type to Python type name."""
    # strip common suffixes for cleaner names
    name = node_type.replace("_", " ").replace("-", " ")
    name = "".join(w.capitalize() for w in name.split())
    # remove common tree-sitter suffixes
    for suffix in ("Node", "Statement", "Expression"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # add "Node" suffix
    return name + "Node"


def find_identifier(node: NodeTS) -> str:
    """recursively find the most relevant identifier descendant of a node."""
    name_field = node.child_by_field_name("name")

    if name_field is not None and name_field.text is not None:
        return name_field.text.decode("utf-8")

    for child in node.named_children:
        if child.type == "identifier" and node.type != "module":
            return child.text.decode("utf-8") if child.text else ""
        if child.child_count > 0:
            result = find_identifier(child)
            if result:
                return result
    return ""


def _module_name_from_source_path(source_file: Path) -> str:
    """extract module name from source file path (e.g. backend/src/services/processing/docling_loader.py -> docling_loader)."""
    return source_file.with_suffix("").stem


LANGUAGE_MODULES = {
    "python": tspython,
    "javascript": tsjs,
    "typescript": tsts,
    "go": tsgo,
    "rust": tsrust,
}

CALL_QUERY_PATTERNS = {
    "python": """
        (call
            function: (identifier) @callee_name
        ) @call_expr
        (call
            function: (attribute
            object: (identifier) @obj
            attribute: (identifier) @callee_name
            )
        ) @call_expr
    """,
    "javascript": """
        (call_expression
            function: (identifier) @callee_name
        ) @call_expr
        (call_expression
            function: (member_expression
            object: (identifier) @obj
            property: (property_identifier) @callee_name
            )
        ) @call_expr
    """,
    "typescript": """
        (call_expression
            function: (identifier) @callee_name
        ) @call_expr
        (call_expression
            function: (member_expression
            object: (identifier) @obj
            property: (property_identifier) @callee_name
            )
        ) @call_expr
    """,
    "go": """
        (call_expression
            function: (identifier) @callee_name
        ) @call_expr
        (call_expression
            function: (selector_expression
            operand: (identifier) @obj
            field: (field_identifier) @callee_name
            )
        ) @call_expr
    """,
    "rust": """
        (call_expression
            function: (identifier) @callee_name
        ) @call_expr
        (call_expression
            function: (field_expression
            value: (identifier) @obj
            field: (field_identifier) @callee_name
            )
        ) @call_expr
    """,
}

FUNCTION_QUERY_PATTERNS = {
    "python": """
        (function_definition
            name: (identifier) @name
            parameters: (parameters) @params
            return_type: (type)? @return_type
            body: (block) @body
        ) @func
    """,
    "javascript": """
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        ) @func
        (variable_declarator
            name: (identifier) @name
            value: (arrow_function) @arrow
        )
    """,
    "typescript": """
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
            return_type: (type_annotation)? @return_type
        ) @func
        (method_definition
            name: (property_identifier) @name
            parameters: (formal_parameters) @params
        ) @method
    """,
    "go": """
        (function_declaration
            name: (identifier) @name
            parameters: (parameter_list) @params
        ) @func
        (method_declaration
            name: (field_identifier) @name
            parameters: (parameter_list) @params
        ) @method
    """,
    "rust": """
        (function_item
            name: (identifier) @name
            parameters: (parameters) @params
        ) @func
    """,
}

CLASS_QUERY_PATTERNS = {
    "python": """
        (class_definition name: (identifier) @name (argument_list (identifier) @base))
    """,
    "javascript": """
        (class_declaration
            name: (identifier) @name
            body: (class_body) @body
            (class_heritage (extends_clause (identifier) @base))?
        ) @class
    """,
    "typescript": """
        (class_declaration
            name: (type_identifier) @name
            body: (class_body) @body
            (class_heritage
            (extends_clause (type_identifier) @extends)?
            (implements_clause (type_identifier) @implements)*
            )?
        ) @class
    """,
    "go": """
        (type_declaration
            (type_spec
            name: (type_identifier) @name
            type: (struct_type) @struct
            )
        ) @class
        (type_declaration
            (type_spec
            name: (type_identifier) @name
            type: (interface_type) @interface
            )
        ) @interface
    """,
    "rust": """
        (struct_item
            name: (type_identifier) @name
            body: (field_declaration_list)? @body
        ) @class
        (enum_item
            name: (type_identifier) @name
        ) @enum
    """,
}


# ── tree-sitter language factory ───────────────────────────────────────────────

LANG_MAP: dict[str, types.ModuleType] = {
    "python": tspython,
    "javascript": tsjs,
    "typescript": tsts,
    "go": tsgo,
    "rust": tsrust,
}


def _get_lang_module(language: str) -> types.ModuleType:
    """get the tree-sitter language module for a given language."""
    lang_mod = LANG_MAP.get(language)
    if lang_mod is None:
        msg = f"Unsupported language: {language}. Supported: {list(LANG_MAP.keys())}"
        raise ValueError(msg)
    return lang_mod


def _get_tree_sitter_lang(language: str) -> ts.Language:
    """get a tree-sitter Language object for the given language."""
    lang_mod = _get_lang_module(language)
    if hasattr(lang_mod, "language"):
        return ts.Language(lang_mod.language())
    if hasattr(lang_mod, "language_typescript"):
        return ts.Language(lang_mod.language_typescript())
    msg = f"Cannot find language getter in {lang_mod}"
    raise ValueError(msg)


_languages: dict[str, ts.Language] = {}
_parsers: dict[str, Parser] = {}


def get_language(language: str) -> ts.Language:
    """Return a cached tree-sitter Language for the given language."""
    if language not in _languages:
        lang_module = LANGUAGE_MODULES.get(language)
        if not lang_module:
            raise ValueError(f"Unsupported language: {language}")
        lang = _get_tree_sitter_lang(language)
        _languages[language] = lang
        _parsers[language] = Parser(lang)
    return _languages[language]

