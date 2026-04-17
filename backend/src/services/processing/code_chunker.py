"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path

import tiktoken
import tree_sitter as ts
import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript
from rich import inspect
from rich.console import Console
from rich.traceback import install
from tree_sitter import Language, Parser

from backend.src.domain.schemas.doc import Subsection
from backend.src.domain.schemas.treesitter_types import COMPOUND as _COMPOUND
from backend.src.domain.schemas.treesitter_types import DEF as _DEF
from backend.src.domain.schemas.treesitter_types import IMPORT as _IMPORT
from backend.src.domain.schemas.treesitter_types import SIMPLE as _SIMPLE
from backend.src.domain.schemas.treesitter_types import T

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


@lru_cache(maxsize=1)
def _get_tiktoken() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model("gpt-4")


def _count_tokens(text: str) -> int:
    return len(_get_tiktoken().encode(text))


def extract_code_chunks(
    language: str,
    source_file: str | None = None,
) -> list[Subsection]:
    """extract code definitions as chunks using tree-sitter."""

    # TODO: implement

    # read source file
    code_path = Path(source_file)  # type: ignore[arg-type]
    code = code_path.read_text(encoding="utf-8")

    # initialize parser with tree-sitter language
    parser = ts.Parser(_get_tree_sitter_lang(language))
    _ = parser.parse(code.encode("utf8"))
    root = _.root_node

    inspect(root)
    breakpoint()

    for node_type in sample_types:
        match node_type:
            case m if m in T.MODULE:
                console.print(f"[blue]module: {node_type}[/blue]")
            case IMPORT if IMPORT in T.IMPORT:
                match node_type:
                    case _IMPORT.DEFAULT:
                        console.print(f"[cyan]import default: {node_type}[/cyan]")
                    case _IMPORT.FROM:
                        console.print(f"[cyan]import from: {node_type}[/cyan]")
                    case _IMPORT.FUTURE:
                        console.print(f"[cyan]import future: {node_type}[/cyan]")
            case CLASS if CLASS in T.CLASS:
                match node_type:
                    case _DEF.CLASS:
                        console.print(f"[green]class: {node_type}[/green]")
                    case _DEF.CLASS_DECORATED:
                        console.print(f"[green]class decorated: {node_type}[/green]")
            case FUNCTION if FUNCTION in T.FUNCTION:
                match node_type:
                    case _DEF.FUNCTION:
                        console.print(f"[green]function: {node_type}[/green]")
            case COMPOUND if COMPOUND in T.COMPOUND:
                match node_type:
                    case _COMPOUND.IF:
                        console.print(f"[yellow]compound if: {node_type}[/yellow]")
                    case _COMPOUND.FOR:
                        console.print(f"[yellow]compound for: {node_type}[/yellow]")
                    case _COMPOUND.WHILE:
                        console.print(f"[yellow]compound while: {node_type}[/yellow]")
                    case _COMPOUND.TRY:
                        console.print(f"[yellow]compound try: {node_type}[/yellow]")
                    case _COMPOUND.WITH:
                        console.print(f"[yellow]compound with: {node_type}[/yellow]")
                    case _COMPOUND.MATCH:
                        console.print(f"[yellow]compound match: {node_type}[/yellow]")
            case SIMPLE if SIMPLE in T.SIMPLE:
                match node_type:
                    case _SIMPLE.EXPRESSION:
                        console.print(f"[magenta]simple expression: {node_type}[/magenta]")
                    case _SIMPLE.RETURN:
                        console.print(f"[magenta]simple return: {node_type}[/magenta]")
                    case _SIMPLE.RAISE:
                        console.print(f"[magenta]simple raise: {node_type}[/magenta]")
                    case _SIMPLE.ASSIGNMENT:
                        console.print(f"[magenta]simple assignment: {node_type}[/magenta]")
                    case _SIMPLE.AUGMENTED_ASSIGNMENT:
                        console.print(f"[magenta]simple augassign: {node_type}[/magenta]")
                    case other:
                        console.print(f"[dim]simple other: {other}[/dim]")
            case _:
 
    # walk the AST, collecting class/function definitions

    # for each node type:
    #   - if class_definition: collect with type "class", recurse into children
    #   - if decorated_definition: look for nested function_definition, recurse
    #   - if function/method/async_definition: collect with inferred type (function/method/async)

    # track seen nodes by start_byte to avoid duplicates

    # sort collected definitions by start_byte

    # for each definition:
    #   - determine end_byte:
    #     - class: find end by dedenting from class line
    #     - function: find block child, use its end_byte
    #   - extract name from identifier/name child
    #   - extract is_async from presence of async child
    #   - extract is_method from def_type == "method"
    #   - call _get_parameters, _get_return_type, _get_docstring, _get_decorators
    #   - slice code from start_byte to end_byte
    #   - count tokens with _count_tokens
    #   - create Subsection with SectionMetadata
    # return list of Subsections
    pass


def _find_class_end(code: str, class_node: ts.Node) -> int:
    """find the end of a class block based on indentation."""
    # TODO: implement
    # get class definition line number from start_byte
    # calculate class line's indentation (leading whitespace)
    # iterate forward through lines:
    #   - skip blank lines
    #   - when line has less or equal indentation to class: return position
    # if no dedent found, return class_node.end_byte
    raise NotImplementedError


def _get_parameters(node: ts.Node) -> list[str]:
    """extract parameter names from function definition."""
    # TODO: implement
    # find parameters/formal_parameters child
    # for each parameter:
    #   - if identifier: add name
    #   - if typed_parameter: find nested identifier, add name
    #   - if self_parameter: add "self"
    # return list of parameter names
    raise NotImplementedError


def _get_return_type(node: ts.Node) -> str | None:
    """extract return type annotation."""
    # TODO: implement
    # find type child
    # return decoded text of type node, or None
    pass


def _get_docstring(node: ts.Node, code: str) -> str | None:
    """extract docstring from function/class node."""
    # TODO: implement
    # find block child
    # if block has children:
    #   - get first child
    #   - if expression_statement, get its first child
    #   - if string/string_start: decode text, strip quotes, return (truncated to 200 chars)
    # return None if no docstring found
    pass


def _get_decorators(node: ts.Node, code: str) -> list[str]:
    """extract decorators from function/class node."""
    # TODO: implement
    # for each child with type "decorator":
    #   - decode text, strip whitespace
    #   - add to result list
    # return result list
    raise NotImplementedError


# --- EXTRACT FUNCTIONS FOR EACH TREE-SITTER TYPE ---


def extract_module(node: ts.Node, code: str) -> str:
    """extract full module content."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_import_statement(node: ts.Node, code: str) -> str:
    """extract import statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_import_from_statement(node: ts.Node, code: str) -> str:
    """extract import from statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_future_import_statement(node: ts.Node, code: str) -> str:
    """extract future import statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_class_definition(node: ts.Node, code: str) -> str:
    """extract class definition."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_decorated_definition(node: ts.Node, code: str) -> str:
    """extract decorated definition (class/function with decorators)."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_function_definition(node: ts.Node, code: str) -> str:
    """extract function definition."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_if_statement(node: ts.Node, code: str) -> str:
    """extract if statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_for_statement(node: ts.Node, code: str) -> str:
    """extract for statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_while_statement(node: ts.Node, code: str) -> str:
    """extract while statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_try_statement(node: ts.Node, code: str) -> str:
    """extract try statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_with_statement(node: ts.Node, code: str) -> str:
    """extract with statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_match_statement(node: ts.Node, code: str) -> str:
    """extract match statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_expression_statement(node: ts.Node, code: str) -> str:
    """extract expression statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_return_statement(node: ts.Node, code: str) -> str:
    """extract return statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_raise_statement(node: ts.Node, code: str) -> str:
    """extract raise statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_assert_statement(node: ts.Node, code: str) -> str:
    """extract assert statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_delete_statement(node: ts.Node, code: str) -> str:
    """extract delete statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_assignment(node: ts.Node, code: str) -> str:
    """extract assignment."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_augmented_assignment(node: ts.Node, code: str) -> str:
    """extract augmented assignment."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_type_alias_statement(node: ts.Node, code: str) -> str:
    """extract type alias statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_yield(node: ts.Node, code: str) -> str:
    """extract yield expression."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_global_statement(node: ts.Node, code: str) -> str:
    """extract global statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_nonlocal_statement(node: ts.Node, code: str) -> str:
    """extract nonlocal statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_break_statement(node: ts.Node, code: str) -> str:
    """extract break statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_continue_statement(node: ts.Node, code: str) -> str:
    """extract continue statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


def extract_pass_statement(node: ts.Node, code: str) -> str:
    """extract pass statement."""
    # TODO: implement
    # return code[node.start_byte : node.end_byte]
    raise NotImplementedError


# =============================================================================
# SUMMARY FACTORY
# Formats metadata + content for embedding
# =============================================================================


def format_for_embedding(
    node_type: str,
    identifier: str,
    parent_chain: list[tuple[str, str]],
    content: str,
    max_content_chars: int = 100,
) -> str:
    """format node metadata and content for embedding.


    Args:
        node_type: tree-sitter node type (e.g., "function_definition", "assignment")
        identifier: node identifier (e.g., function name, variable name)
        parent_chain: list of (type, identifier) tuples from root to parent
        content: the actual code content
        max_content_chars: truncate content beyond this length

    Returns:
        Formatted string: metadata context + content for embedding
    """
    # build parent path: "MODULE > CLASS > METHOD"
    path_parts = [t for t, _ in parent_chain]
    if identifier:
        path_parts.append(f"{node_type}:{identifier}")
    else:
        path_parts.append(node_type)

    parent_path = " > ".join(path_parts)

    # truncate content
    truncated_content = (
        content[:max_content_chars] + "..."
        if len(content) > max_content_chars
        else content
    )

    return f"{parent_path}\n{truncated_content}"


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
