from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from backend.src.settings import EmbedderConfig

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tsts
from code_chunker import (
    ChunkerConfig,
    ChunkType,
    CodeChunk,
    CodeChunker,
    Import,
    ParseResult,
)
from tree_sitter import Language, Parser, Query, QueryCursor

from backend.src.domain.enums import GraphNodeType
from backend.src.domain.schemas.graph import (
    CallEdge,
    CodeNode,
    GraphResult,
    InheritanceEdge,
    MethodEdge,
    Parameter,
    ResolvedImport,
    Signature,
)

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
        (call_expression
          function: (selector_expression
            operand: (pkg_import) @pkg
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
        (call_expression
          function: (scoped_identifier
            path: (identifier) @mod
            name: (type_identifier) @callee_name
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
          result: (parameter_list)? @return_type
        ) @func
        (method_declaration
          name: (field_identifier) @name
          parameters: (parameter_list) @params
          receiver: (parameter_list) @receiver
        ) @method
    """,
    "rust": """
        (function_item
          name: (identifier) @name
          parameters: (parameters) @params
          return_type: (type)? @return_type
        ) @func
    """,
}

CLASS_QUERY_PATTERNS = {
    "python": """
        (class_definition
          name: (identifier) @name
          body: (block) @body
          superclasses: (argument_list)? @bases
        ) @class
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


class CallGraphService:
    _parsers: dict[str, Parser] = {}
    _languages: dict[str, Language] = {}

    @classmethod
    def _get_parser(cls, language: str) -> Parser:
        if language not in cls._parsers:
            lang_module = LANGUAGE_MODULES.get(language)
            if not lang_module:
                raise ValueError(f"Unsupported language: {language}")
            if language == "typescript":
                lang = Language(lang_module.language_typescript())
            else:
                lang = Language(lang_module.language())
            cls._languages[language] = lang
            cls._parsers[language] = Parser(lang)
        return cls._parsers[language]

    @classmethod
    def _get_language(cls, language: str) -> Language:
        if language not in cls._languages:
            cls._get_parser(language)
        return cls._languages[language]

    def extract_graph(
        self,
        source_code: str,
        file_path: str,
        file_id: str,
        language: str,
    ) -> GraphResult:
        parser = self._get_parser(language)
        tree = parser.parse(source_code.encode())
        root = tree.root_node

        code_chunker = CodeChunker(
            config=ChunkerConfig(
                include_comments=False,
                include_imports=True,
            )
        )

        parse_result = code_chunker.parse(source_code, language)

        nodes = self._extract_nodes(
            root, source_code, file_path, file_id, language, parse_result
        )

        call_edges = self._extract_call_edges(
            root, source_code, file_path, file_id, language, nodes
        )

        inheritance_edges = self._extract_inheritance(
            root, source_code, file_path, file_id, language, nodes
        )

        imports = self._extract_imports(parse_result, file_id)

        return GraphResult(
            file_id=file_id,
            file_path=file_path,
            language=language,
            nodes=nodes,
            call_edges=call_edges,
            inheritance_edges=inheritance_edges,
            imports=imports,
        )

    def _extract_nodes(
        self,
        root: Any,
        source_code: str,
        file_path: str,
        file_id: str,
        language: str,
        parse_result: ParseResult,
    ) -> list[CodeNode]:
        nodes = []
        lang = self._get_language(language)

        chunk_map = {c.name: c for c in parse_result.chunks if c.name}

        # Extract classes FIRST to build class membership map before function extraction
        class_info: dict[
            int, tuple[str, int, int]
        ] = {}  # line_start -> (name, line_start, line_end)
        class_query = Query(lang, CLASS_QUERY_PATTERNS.get(language, ""))
        class_cursor = QueryCursor(class_query)

        for pattern_idx, captures in class_cursor.matches(root):
            if "name" in captures:
                for name_node in captures["name"]:
                    class_node = name_node.parent
                    if not class_node:
                        continue

                    name = source_code[name_node.start_byte : name_node.end_byte]
                    line_start = name_node.start_point[0] + 1
                    line_end = class_node.end_point[0] + 1

                    class_info[line_start] = (name, line_start, line_end)
                    nodes.append(
                        CodeNode(
                            node_type=GraphNodeType.CLASS,
                            name=name,
                            qualified_name=f"{file_path}:{name}",
                            file_id=file_id,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            source_code=source_code[
                                class_node.start_byte : class_node.end_byte
                            ],
                        )
                    )

        # Find containing class for a given line number
        def find_enclosing_class(line: int) -> str | None:
            # class_info keys are sorted class starting lines
            for cls_line in sorted(class_info.keys(), reverse=True):
                if line >= cls_line:
                    cls_name, cls_start, cls_end = class_info[cls_line]
                    if cls_start <= line <= cls_end:
                        return cls_name
                    break
            return None

        # Extract functions and methods
        func_query = Query(lang, FUNCTION_QUERY_PATTERNS.get(language, ""))
        func_cursor = QueryCursor(func_query)

        for pattern_idx, captures in func_cursor.matches(root):
            if "name" in captures:
                name_nodes = captures["name"]
                for name_node in name_nodes:
                    func_node = name_node.parent
                    if not func_node:
                        continue

                    name = source_code[name_node.start_byte : name_node.end_byte]
                    line_start = name_node.start_point[0] + 1
                    line_end = func_node.end_point[0] + 1

                    is_method = self._is_method(func_node, language)
                    containing_class = (
                        find_enclosing_class(line_start) if is_method else None
                    )

                    node_type = (
                        GraphNodeType.METHOD if is_method else GraphNodeType.FUNCTION
                    )

                    if containing_class:
                        qualified_name = f"{file_path}:{containing_class}.{name}"
                    else:
                        qualified_name = f"{file_path}:{name}"

                    signature = self._extract_signature(
                        func_node, source_code, language
                    )

                    nodes.append(
                        CodeNode(
                            node_type=node_type,
                            name=name,
                            qualified_name=qualified_name,
                            file_id=file_id,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            signature=signature,
                            docstring=self._extract_docstring(
                                func_node, source_code, language
                            ),
                            source_code=source_code[
                                func_node.start_byte : func_node.end_byte
                            ],
                            metadata={"containing_class": containing_class}
                            if containing_class
                            else {},
                        )
                    )

        nodes.append(
            CodeNode(
                node_type=GraphNodeType.FILE,
                name=Path(file_path).name,
                qualified_name=file_path,
                file_id=file_id,
                file_path=file_path,
                line_start=1,
                line_end=source_code.count("\n") + 1,
            )
        )

        return nodes

    def _extract_signature(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> Signature:
        params = self._extract_parameters(func_node, source_code, language)
        return_type = self._extract_return_type(func_node, source_code, language)
        is_async = self._is_async(func_node, language)
        receiver = self._extract_receiver(func_node, source_code, language)
        decorators = self._extract_decorators(func_node, source_code, language)

        return Signature(
            parameters=params,
            return_type=return_type,
            is_async=is_async,
            is_method=receiver is not None,
            receiver=receiver,
            decorators=decorators,
        )

    def _extract_parameters(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> list[Parameter]:
        params = []

        if language == "python":
            params_node = func_node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.append(
                            Parameter(
                                name=source_code[child.start_byte : child.end_byte],
                            )
                        )
                    elif child.type == "typed_parameter":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        default_node = child.child_by_field_name("default")

                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "",
                                type_annotation=source_code[
                                    type_node.start_byte : type_node.end_byte
                                ]
                                if type_node
                                else None,
                                default_value=source_code[
                                    default_node.start_byte : default_node.end_byte
                                ]
                                if default_node
                                else None,
                            )
                        )
                    elif child.type == "default_parameter":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "",
                                default_value=source_code[
                                    value_node.start_byte : value_node.end_byte
                                ]
                                if value_node
                                else None,
                            )
                        )
                    elif child.type in (
                        "list_splat_pattern",
                        "dictionary_splat_pattern",
                    ):
                        name_node = child.child_by_field_name("name")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "*args"
                                if child.type == "list_splat_pattern"
                                else "**kwargs",
                                is_variadic=child.type == "list_splat_pattern",
                                is_keyword=child.type == "dictionary_splat_pattern",
                            )
                        )

        elif language in ("javascript", "typescript"):
            params_node = func_node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.append(
                            Parameter(
                                name=source_code[child.start_byte : child.end_byte],
                            )
                        )
                    elif child.type == "assignment_pattern":
                        left = child.child_by_field_name("left")
                        right = child.child_by_field_name("right")
                        if left:
                            params.append(
                                Parameter(
                                    name=source_code[left.start_byte : left.end_byte],
                                    default_value=source_code[
                                        right.start_byte : right.end_byte
                                    ]
                                    if right
                                    else None,
                                )
                            )
                    elif child.type == "rest_pattern":
                        name_node = child.child_by_field_name("name")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "...",
                                is_variadic=True,
                            )
                        )
                    elif child.type == "required_parameter":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "",
                                type_annotation=source_code[
                                    type_node.start_byte : type_node.end_byte
                                ]
                                if type_node
                                else None,
                            )
                        )
                    elif child.type == "optional_parameter":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        default_node = child.child_by_field_name("default")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "",
                                type_annotation=source_code[
                                    type_node.start_byte : type_node.end_byte
                                ]
                                if type_node
                                else None,
                                default_value=source_code[
                                    default_node.start_byte : default_node.end_byte
                                ]
                                if default_node
                                else None,
                            )
                        )

        elif language == "go":
            params_node = func_node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "parameter_declaration":
                        names = []
                        type_node = None
                        for sub in child.children:
                            if sub.type == "identifier":
                                names.append(source_code[sub.start_byte : sub.end_byte])
                            elif sub.type in (
                                "type_identifier",
                                "qualified_type",
                                "pointer_type",
                                "slice_type",
                                "array_type",
                                "map_type",
                                "channel_type",
                                "function_type",
                                "interface_type",
                                "struct_type",
                            ):
                                type_node = sub
                        type_str = (
                            source_code[type_node.start_byte : type_node.end_byte]
                            if type_node
                            else None
                        )
                        for name in names:
                            params.append(
                                Parameter(name=name, type_annotation=type_str)
                            )

        elif language == "rust":
            params_node = func_node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "parameter":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        params.append(
                            Parameter(
                                name=source_code[
                                    name_node.start_byte : name_node.end_byte
                                ]
                                if name_node
                                else "",
                                type_annotation=source_code[
                                    type_node.start_byte : type_node.end_byte
                                ]
                                if type_node
                                else None,
                            )
                        )

        return params

    def _extract_return_type(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> str | None:
        type_node = func_node.child_by_field_name("return_type")
        if type_node:
            return source_code[type_node.start_byte : type_node.end_byte]

        if language == "go":
            result = func_node.child_by_field_name("result")
            if result:
                return source_code[result.start_byte : result.end_byte]

        return None

    def _is_async(self, func_node: Any, language: str) -> bool:
        if language == "python":
            for child in func_node.children:
                if child.type == "async":
                    return True
            return False
        if language in ("javascript", "typescript"):
            for child in func_node.children:
                if child.type == "async":
                    return True
        return False

    def _is_method(self, func_node: Any, language: str) -> bool:
        if language == "python":
            parent = func_node.parent
            return parent is not None and parent.type == "class_definition"
        if language == "go":
            receiver = func_node.child_by_field_name("receiver")
            return receiver is not None
        if language in ("javascript", "typescript"):
            parent = func_node.parent
            return parent is not None and parent.type == "class_body"
        return False

    def _extract_receiver(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> str | None:
        if language == "go":
            receiver = func_node.child_by_field_name("receiver")
            if receiver:
                for child in receiver.children:
                    if child.type == "parameter_declaration":
                        type_node = None
                        for sub in child.children:
                            if sub.type in (
                                "type_identifier",
                                "pointer_type",
                                "qualified_type",
                            ):
                                type_node = sub
                                break
                        if type_node:
                            return source_code[
                                type_node.start_byte : type_node.end_byte
                            ]
        return None

    def _extract_decorators(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> list[str]:
        decorators = []
        if language == "python":
            parent = func_node.parent
            if parent:
                for i, child in enumerate(parent.children):
                    if child == func_node:
                        for prev in parent.children[:i]:
                            if prev.type == "decorator":
                                decorators.append(
                                    source_code[prev.start_byte : prev.end_byte]
                                )
        elif language in ("javascript", "typescript"):
            for child in func_node.children:
                if child.type == "decorator":
                    decorators.append(source_code[child.start_byte : child.end_byte])
        return decorators

    def _extract_docstring(
        self,
        func_node: Any,
        source_code: str,
        language: str,
    ) -> str | None:
        if language == "python":
            body = func_node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "expression_statement":
                        expr = child.child(0)
                        if expr and expr.type == "string":
                            doc = source_code[expr.start_byte : expr.end_byte]
                            return doc.strip("\"'")
        return None

    def _extract_call_edges(
        self,
        root: Any,
        source_code: str,
        file_path: str,
        file_id: str,
        language: str,
        nodes: list[CodeNode],
    ) -> list[CallEdge]:
        edges = []
        lang = self._get_language(language)

        query = Query(lang, CALL_QUERY_PATTERNS.get(language, ""))

        func_map = {}
        for node in nodes:
            if node.node_type in (GraphNodeType.FUNCTION, GraphNodeType.METHOD):
                func_map[node.line_start] = node

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
                        edges.append(
                            CallEdge(
                                caller_name=caller_node.name,
                                caller_qualified=caller_node.qualified_name,
                                caller_file_id=file_id,
                                callee_name=callee_name,
                                callee_qualified=None,
                                callee_file_id=None,
                                call_expression=call_expression,
                                line_number=line_number,
                                confidence=0.7,
                                is_cross_file=False,
                            )
                        )

        return edges

    def _extract_inheritance(
        self,
        root: Any,
        source_code: str,
        file_path: str,
        file_id: str,
        language: str,
        nodes: list[CodeNode],
    ) -> list[InheritanceEdge]:
        edges = []
        lang = self._get_language(language)

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

                        child_name = source_code[
                            name_node.start_byte : name_node.end_byte
                        ]
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

    def _extract_imports(
        self,
        parse_result: ParseResult,
        file_id: str,
    ) -> list[ResolvedImport]:
        imports = []
        for imp in parse_result.imports:
            imports.append(
                ResolvedImport(
                    import_statement=f"from {imp.module} import {', '.join(imp.names)}"
                    if imp.names
                    else f"import {imp.module}",
                    imported_names=imp.names or [imp.module],
                    source_file_id=file_id,
                    target_file_id=None,
                    target_file_path=None,
                    line_number=imp.line_number,
                    is_stdlib=self._is_stdlib(imp.module),
                    is_third_party=self._is_third_party(imp.module),
                )
            )
        return imports

    def _is_stdlib(self, module: str) -> bool:
        stdlib_prefixes = {
            "os",
            "sys",
            "json",
            "re",
            "time",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "typing",
            "asyncio",
            "concurrent",
            "threading",
            "multiprocessing",
            "logging",
            "argparse",
            "configparser",
            "io",
            "pickle",
            "struct",
            "codecs",
            "csv",
            "hashlib",
            "hmac",
            "secrets",
            "random",
            "statistics",
            "math",
            "cmath",
            "decimal",
            "fractions",
            "copy",
            "pprint",
            "reprlib",
            "enum",
            "graphlib",
            "operator",
            "string",
            "textwrap",
            "unicodedata",
            "difflib",
            "heapq",
            "bisect",
            "array",
            "weakref",
            "types",
            "contextlib",
            "abc",
            "dataclasses",
            "traceback",
            "warnings",
            "__future__",
        }
        return module.split(".")[0] in stdlib_prefixes

    def _is_third_party(self, module: str) -> bool:
        return not self._is_stdlib(module) and not module.startswith(".")
