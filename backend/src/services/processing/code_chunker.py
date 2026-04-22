"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

import types
from pathlib import Path

import tree_sitter as ts
import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python  # type: ignore
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript
from rich.pretty import pprint as pp
from tree_sitter import (
    Language,  # type: ignore
    Parser,  # type: ignore
)
from tree_sitter import Node as NodeTS

from backend.src.domain.schemas.treesitter_types import (
    Node,
    Range,
)
from backend.src.services.processing.utils.parse_utils import (
    _module_name_from_source_path,
)
from backend.src.services.processing.utils.parsing_funcs import (
    parse_node,
)

CLAUSE_TYPES = {"else_clause", "elif_clause", "case_clause"}

# annotate as ModuleType to allow attribute access for .language()
ts_parser = tstypescript.language_typescript()
tsx_parser = tstypescript.language_tsx()
LANG_MAP: dict[str, types.ModuleType] = {
    "python": tspython,
    "javascript": tsjs,
    "typescript": tstypescript,
    "go": tsgo,
    "rust": tsrust,
}

LANG_NAMES: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
}


def _get_lang_module(language: str) -> types.ModuleType:
    """get the tree-sitter language module for a given language."""
    language = LANG_NAMES.get(language.lower(), language.lower())
    lang_mod = LANG_MAP.get(language)
    if lang_mod is None:
        msg = f"Unsupported language: {language}. Supported: {list(LANG_MAP.keys())}"
        raise ValueError(msg)
    return lang_mod


def _get_tree_sitter_lang(language: str) -> ts.Language:
    """get a tree-sitter Language object for the given language."""
    lang_mod = _get_lang_module(language)
    # different packages expose different api methods
    if hasattr(lang_mod, "language"):
        return ts.Language(lang_mod.language())
    if hasattr(lang_mod, "language_typescript"):
        return ts.Language(lang_mod.language_typescript())
    msg = f"Cannot find language getter in {lang_mod}"
    raise ValueError(msg)


def parse_file(source_file: Path) -> None:
    """test parse_node / parse_compound_statement on a source file."""
    python_lang = Language(tree_sitter_python.language())

    parser = Parser(python_lang)

    code = source_file.read_text()
    tree = parser.parse(bytes(code, "utf-8"))
    root = tree.root_node

    module_name = _module_name_from_source_path(source_file)

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
    )

    def parse_tree(node: NodeTS, parent: Node) -> Node:
        """Recursively parse tree-sitter node into our Node structure."""
        node_type = node.type

        # Skip the root module node itself — its children become our root's children
        if node_type == "module":
            for child in node.named_children:
                parsed = parse_node(child, parent=parent)
                if parsed is not None:
                    parent.children.append(parsed)
            return parent

        # For all other nodes, parse them
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
                # Parse children into body statements using parsed as parent for scoping
                if body.statements:
                    parsed.children = list(body.statements)

        return parent

    result = parse_tree(root, parent=root_node)

    pp(result)


# =============================================================================
# SUMMARY FACTORY
# Formats metadata + content for embedding
# =============================================================================


def parse_hierarchical_id(node_id: str) -> dict[str, str | int | None]:
    """parse a hierarchical node ID into its components.

    Args:
        node_id: hierarchical ID like "file.py:Parser:_get_parser:stmt_0:expr_0"

    Returns:
        dict with parsed components:
            - file_path: str
            - class_name: str | None
            - function_name: str | None
            - statement_index: int | None
            - expression_index: int | None
    """
    # TODO: implement
    parts = node_id.split(":")
    return {
        "file_path": parts[0] if len(parts) > 0 else None,
        "class_name": parts[1] if len(parts) > 1 else None,
        "function_name": parts[2] if len(parts) > 2 else None,
        "statement_index": int(parts[3])
        if len(parts) > 3 and parts[3].isdigit()
        else None,
        "expression_index": int(parts[4])
        if len(parts) > 4 and parts[4].isdigit()
        else None,
    }


def build_hierarchical_id(
    file_path: str,
    class_name: str | None = None,
    function_name: str | None = None,
    statement_index: int | None = None,
    expression_index: int | None = None,
) -> str:
    """build a hierarchical node ID.

    Args:
        file_path: source file path
        class_name: containing class name (if any)
        function_name: containing function name (if any)
        statement_index: index within parent block
        expression_index: index within statement

    Returns:
        hierarchical ID string
    """
    # TODO: implement
    parts = [file_path]
    if class_name:
        parts.append(class_name)
    if function_name:
        parts.append(function_name)
    if statement_index is not None:
        parts.append(f"stmt_{statement_index}")
    if expression_index is not None:
        parts.append(f"expr_{expression_index}")
    return ":".join(parts)


def get_sibling_position(
    parent_children: list[str],
    child_id: str,
) -> tuple[int, int]:
    """get position of child within parent's children.

    Args:
        parent_children: list of child IDs in order
        child_id: ID of child to find position for

    Returns:
        tuple of (index, total) where index is 1-based
    """
    # TODO: implement
    try:
        index = parent_children.index(child_id) + 1  # 1-based
    except ValueError:
        index = 0
    return (index, len(parent_children))
