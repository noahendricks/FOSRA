"""tree-sitter based code chunker with language-aware parsing."""

from __future__ import annotations

import types
from functools import lru_cache
from typing import Literal

import tiktoken
import tree_sitter as ts
import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript

from backend.src.domain.schemas.doc import Subsection, SectionMetadata

# annotate as ModuleType to allow attribute access for .language()
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
    code: str,
    language: str,
    source_file: str | None = None,
) -> list[Subsection]:
    """extract code definitions as chunks using tree-sitter."""
    parser = ts.Parser(_get_tree_sitter_lang(language))
    tree = parser.parse(code.encode("utf8"))

    defs: list[
        tuple[ts.Node, Literal["class", "function", "method", "async_function"]]
    ] = []
    seen: set[int] = set()

    def walk(node: ts.Node, parent_type: str | None = None) -> None:
        """walk AST and collect function/class/method definitions."""
        if node.start_byte in seen:
            return
        seen.add(node.start_byte)

        if node.type == "class_definition":
            defs.append((node, "class"))
            for child in node.children:
                walk(child, "class")
            return

        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    walk(child, parent_type)
            return

        if node.type in (
            "function_definition",
            "method_definition",
            "async_function_definition",
        ):
            is_method = parent_type == "class"
            is_async = any(c.type == "async" for c in node.children)
            def_type: Literal["class", "function", "method", "async_function"] = (
                "async_function"
                if is_async
                else ("method" if is_method else "function")
            )
            defs.append((node, def_type))

        for child in node.children:
            walk(child, parent_type)

    walk(tree.root_node)
    defs.sort(key=lambda x: x[0].start_byte)

    chunks = []
    for node, def_type in defs:
        start_byte = node.start_byte
        end_byte = node.end_byte

        if node.type == "class_definition":
            end_byte = _find_class_end(code, node)
        else:
            for child in node.children:
                if child.type == "block":
                    end_byte = child.end_byte
                    break

        name = None
        for child in node.children:
            if child.type in ("identifier", "name"):
                text_bytes = child.text
                if text_bytes is not None:
                    name = text_bytes.decode("utf-8")
                break

        is_async = any(c.type == "async" for c in node.children)
        is_method = def_type == "method"
        params = _get_parameters(node)
        return_type = _get_return_type(node)
        docstring = _get_docstring(node, code)
        decorators = _get_decorators(node, code)

        text = code[start_byte:end_byte]
        token_count = _count_tokens(text)

        chunks.append(
            Subsection(
                text=text,
                metadata=SectionMetadata(
                    section_id=name or f"chunk_{start_byte}",
                    start_char=start_byte,
                    end_char=end_byte,
                    token_count=token_count,
                    source_file=source_file,
                    code_definition_type=def_type,
                    is_async=is_async,
                    is_method=is_method,
                    decorators=decorators if decorators else None,
                    docstring=docstring,
                    parameters=params if params else None,
                    return_type=return_type,
                ),
            )
        )

    return chunks


def _find_class_end(code: str, class_node: ts.Node) -> int:
    """find the end of a class block based on indentation."""
    lines = code.split("\n")
    class_line_no = code[: class_node.start_byte].count("\n")
    class_line = lines[class_line_no]
    class_indent = len(class_line) - len(class_line.lstrip())

    for i in range(class_line_no + 1, len(lines)):
        line = lines[i]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= class_indent:
                return code.index(lines[i], class_node.start_byte)
    return class_node.end_byte


def _get_parameters(node: ts.Node) -> list[str]:
    """extract parameter names from function definition."""
    params: list[str] = []
    for child in node.children:
        if child.type in ("parameters", "formal_parameters"):
            for p in child.children:
                if p.type in ("identifier", "typed_parameter"):
                    for c in p.children:
                        if c.type == "identifier":
                            text_bytes = c.text
                            if text_bytes is not None:
                                params.append(text_bytes.decode("utf-8"))
                elif p.type in ("self_parameter",):
                    params.append("self")
    return params


def _get_return_type(node: ts.Node) -> str | None:
    """extract return type annotation."""
    for child in node.children:
        if child.type == "type":
            text_bytes = child.text
            if text_bytes is not None:
                return text_bytes.decode("utf-8", errors="ignore")
    return None


def _get_docstring(node: ts.Node, code: str) -> str | None:
    """extract docstring from function/class node."""
    for child in node.children:
        if child.type == "block" and child.children:
            first = child.children[0]
            if first.type == "expression_statement":
                first = first.children[0]
            if first.type in ("string", "string_start"):
                text_bytes = first.text
                if text_bytes is not None:
                    text = text_bytes.decode("utf-8", errors="ignore")
                    if text.strip().startswith(('"', "'", '"""', "'''")):
                        return text.strip("\"'").strip()[:200]
    return None


def _get_decorators(node: ts.Node, code: str) -> list[str]:
    """extract decorators from function/class node."""
    result: list[str] = []
    for c in node.children:
        if c.type == "decorator":
            text_bytes = c.text
            if text_bytes is not None:
                result.append(text_bytes.decode("utf-8", errors="ignore").strip())
    return result
