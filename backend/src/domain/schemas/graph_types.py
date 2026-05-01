from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import tree_sitter as ts

from backend.src.storage.utils.converters import DomainStruct

if TYPE_CHECKING:
    from backend.src.domain.schemas.retrieval import AccumulatedItem


# =============================================================================
# Graph node types
# =============================================================================


class GraphNodeType(StrEnum):
    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    CONSTANT = "Constant"
    IMPORT = "Import"
    TYPE_ALIAS = "TypeAlias"
    IF = "IfStatement"
    FOR = "ForStatement"
    WHILE = "WhileStatement"
    TRY = "TryStatement"
    WITH = "WithStatement"
    MATCH = "MatchStatement"


# =============================================================================
# Imports
# =============================================================================


class IMPORT(StrEnum):
    DEFAULT = "import_statement"
    FROM = "import_from_statement"
    FUTURE = "future_import_statement"


IMPORT_TYPES: set[str] = {IMPORT.DEFAULT, IMPORT.FROM, IMPORT.FUTURE}


# =============================================================================
# Class/Function
# =============================================================================


class DEF(StrEnum):
    CLASS = "class_definition"
    CLASS_DECORATED = "decorated_definition"
    FUNCTION = "function_definition"


CLASS_TYPES: set[str] = {DEF.CLASS}
FUNCTION_TYPES: set[str] = {DEF.FUNCTION}
DEFINITION_TYPES: set[str] = CLASS_TYPES | FUNCTION_TYPES


# =============================================================================
# Module-level definitions
# =============================================================================


class MODULE_DEF(StrEnum):
    CONSTANT = "assignment"
    TYPE_ALIAS = "type_alias_statement"


MODULE_DEF_TYPES: set[str] = {MODULE_DEF.CONSTANT, MODULE_DEF.TYPE_ALIAS}


# =============================================================================
# Compound statements
# =============================================================================


class COMPOUND(StrEnum):
    IF = "if_statement"
    FOR = "for_statement"
    WHILE = "while_statement"
    TRY = "try_statement"
    WITH = "with_statement"
    MATCH = "match_statement"


COMPOUND_TYPES: set[str] = {
    COMPOUND.IF,
    COMPOUND.FOR,
    COMPOUND.WHILE,
    COMPOUND.TRY,
    COMPOUND.WITH,
    COMPOUND.MATCH,
}


# =============================================================================
# Simple statements
# =============================================================================


class SIMPLE(StrEnum):
    RETURN = "return_statement"
    RAISE = "raise_statement"
    ASSERT = "assert_statement"
    DELETE = "delete_statement"
    ASSIGNMENT = "assignment"
    AUGMENTED_ASSIGNMENT = "augmented_assignment"
    TYPE_ALIAS = "type_alias_statement"
    YIELD = "yield"
    GLOBAL = "global_statement"
    NONLOCAL = "nonlocal_statement"
    BREAK = "break_statement"
    CONTINUE = "continue_statement"
    PASS = "pass_statement"


SIMPLE_TYPES: set[str] = {
    SIMPLE.RETURN,
    SIMPLE.RAISE,
    SIMPLE.ASSERT,
    SIMPLE.DELETE,
    SIMPLE.ASSIGNMENT,
    SIMPLE.AUGMENTED_ASSIGNMENT,
    SIMPLE.TYPE_ALIAS,
    SIMPLE.YIELD,
    SIMPLE.GLOBAL,
    SIMPLE.NONLOCAL,
    SIMPLE.BREAK,
    SIMPLE.CONTINUE,
    SIMPLE.PASS,
}
COMMENT_TYPES: set[str] = {"comment"}
STATEMENT_TYPES: set[str] = IMPORT_TYPES | SIMPLE_TYPES | COMMENT_TYPES


# =============================================================================
# Expressions
# =============================================================================


class EXPR(StrEnum):
    CALL = "call"
    ATTRIBUTE = "attribute"
    SUBSCRIPT = "subscript"
    BINARY = "binary_operator"
    COMPARISON = "comparison_operator"
    BOOLEAN = "boolean_operator"
    CONDITIONAL = "conditional_expression"
    LAMBDA = "lambda"
    NAMED = "named_expression"
    LIST = "list_literal"
    TUPLE = "tuple_literal"
    SET = "set_literal"
    DICT = "dictionary"
    DICT_COMP = "dictionary_comprehension"
    GENERATOR = "generator_expression"
    AWAIT = "await"
    UNARY = "unary_operator"
    CONCAT_STRING = "concatenated_string"
    ARG_LIST = "argument_list"
    KW_ARG = "keyword_argument"
    DOTTED_NAME = "dotted_name"
    IDENTIFIER = "identifier"


EXPRESSION_TYPES: set[str] = {
    EXPR.CALL,
    EXPR.ATTRIBUTE,
    EXPR.SUBSCRIPT,
    EXPR.BINARY,
    EXPR.COMPARISON,
    EXPR.BOOLEAN,
    EXPR.CONDITIONAL,
    EXPR.LAMBDA,
    EXPR.NAMED,
    EXPR.LIST,
    EXPR.TUPLE,
    EXPR.SET,
    EXPR.DICT,
    EXPR.DICT_COMP,
    EXPR.GENERATOR,
    EXPR.AWAIT,
    EXPR.UNARY,
    EXPR.CONCAT_STRING,
    EXPR.ARG_LIST,
    EXPR.KW_ARG,
    EXPR.DOTTED_NAME,
}


LITERAL_TYPES: set[str] = {
    "string",
    "integer",
    "float",
    "true",
    "false",
    "none",
}

STRUCTURAL_TYPES: set[str] = {
    "identifier",
    "type_parameter",
    "dotted_name",
    "keyword_identifier",
    "slice",
    "ellipsis",
    "pattern_list",
    "generic_type",
    "union_type",
}

PATTERN_TYPES: set[str] = {
    "class_pattern",
    "dict_pattern",
    "list_pattern",
    "tuple_pattern",
}


# =============================================================================
# Structural helpers
# =============================================================================


def is_import(node_type: str) -> bool:
    return node_type in IMPORT_TYPES


def is_class(node_type: str) -> bool:
    return node_type in CLASS_TYPES


def is_function(node_type: str) -> bool:
    return node_type in FUNCTION_TYPES


def is_definition(node_type: str) -> bool:
    return node_type in DEFINITION_TYPES


def is_compound(node_type: str) -> bool:
    return node_type in COMPOUND_TYPES


def is_simple(node_type: str) -> bool:
    return node_type in SIMPLE_TYPES


def is_statement(node_type: str) -> bool:
    return node_type in STATEMENT_TYPES


def is_expression(node_type: str) -> bool:
    return node_type in EXPRESSION_TYPES


def is_decorated(node_type: str) -> bool:
    return node_type == DEF.CLASS_DECORATED


def is_dotted_name(node_type: str) -> bool:
    return node_type == "dotted_name"


# =============================================================================
# Tree-sitter type enums and constants (for parsing)
# =============================================================================


class MODULE(StrEnum):
    MODULE = "module"


class COMMENT(StrEnum):
    COMMENT = "comment"

# Literals
class LITERAL(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    TRUE = "true"
    FALSE = "false"
    NONE = "none"


# Structural - building blocks for imports, patterns, etc.
class STRUCT(StrEnum):
    IDENTIFIER = "identifier"
    TYPE_PARAM = "type_parameter"
    DOTTED_NAME = "dotted_name"
    KW_IDENTIFIER = "keyword_identifier"
    SLICE = "slice"
    ELLIPSIS = "ellipsis"
    PATTERN_LIST = "pattern_list"
    GENERIC_TYPE = "generic_type"
    UNION_TYPE = "union_type"



# Pattern types - for match statements (Python 3.10+)
class PATTERN(StrEnum):
    CLASS = "class_pattern"
    DICT = "dict_pattern"
    LIST = "list_pattern"
    TUPLE = "tuple_pattern"




# =============================================================================
# Domain types (graph-ready)
# =============================================================================


class Signature(DomainStruct):
    """function signature with parameters and return type."""

    parameters: "Parameters | None" = None
    return_type: str | None = None
    is_async: bool = False
    is_method: bool = False
    receiver: str | None = None
    decorators: list[str] = []


class ResolvedImport(DomainStruct, kw_only=True):
    """an import statement resolved to its target file."""

    import_statement: str
    imported_names: list[str]
    source_file_id: str
    target_file_id: str | None = None
    target_file_path: str | None = None
    line_number: int


class CallEdge(DomainStruct, kw_only=True):
    """represents a call from one function/method to another."""

    caller_name: str
    caller_qualified: str
    caller_file_id: str
    callee_name: str
    callee_qualified: str | None = None
    callee_file_id: str | None = None
    call_expression: str
    line_number: int
    confidence: float = 1.0
    is_cross_file: bool = False


class InheritanceEdge(DomainStruct):
    """represents a class inheritance/implementation relationship."""

    child_name: str
    child_qualified: str
    child_file_id: str
    parent_name: str
    parent_qualified: str | None = None
    parent_file_id: str | None = None
    inheritance_type: str = "extends"
    is_cross_file: bool = False


class MethodEdge(DomainStruct):
    """represents a class defining/containing a method."""

    class_name: str
    class_qualified: str
    class_file_id: str
    method_name: str
    method_qualified: str
    method_file_id: str


class FunctionMetadata(DomainStruct):
    """domain model for function/method metadata extracted from AST nodes."""

    containing_class: str | None = None
    is_public: bool = True
    is_private: bool = False
    is_property: bool = False
    is_coroutine: bool = False
    is_abstract: bool = False
    is_override: bool = False
    is_deprecated: bool = False
    decorators: list[str] = []
    param_count: int = 0
    is_static: bool = False
    is_overload: bool = False


class TypeAliasMetadata(DomainStruct):
    type_expr: str
    is_public: bool = False


class ClassMetadata(DomainStruct):
    superclasses: list[str] = []
    is_public: bool = False
    is_private: bool = False
    is_abstract: bool = False
    is_exception: bool = False
    decorators: list[str] = []
    method_count: int = 0
    is_dataclass: bool = False
    is_enum: bool = False


class ConstantMetadata(DomainStruct):
    value_repr: str | None = None
    value_type: str | None = None
    is_public: bool = False
    is_private: bool = False


class ImportMetadata(DomainStruct):
    """Metadata extracted from an import node."""

    type: str = IMPORT.DEFAULT
    from_names: list[str] = []
    target_file_id: str | None = None
    target_file_path: str | None = None
    source_file_id: str = ""
    imported_names: list[str] = []
    line_number: int | None = None
    aliased: str | None = None
    is_relative: bool = False
    is_wildcard: bool = False


class ControlFlowMetadata(DomainStruct):
    """Metadata for if/for/while/try/with/match statements."""

    condition: str | None = None
    target: str | None = None
    iterable: str | None = None
    exception_type: str | None = None
    alias: str | None = None
    guard: str | None = None


# =============================================================================
# CodeNode — graph-ready node for all code elements
# =============================================================================


class CodeNode(DomainStruct):
    """a node in the code graph (file, module, class, function, method)."""

    node_type: GraphNodeType
    name: str
    qualified_name: str
    file_id: str
    file_path: str
    line_start: int
    line_end: int
    metadata: (
        FunctionMetadata
        | ClassMetadata
        | ImportMetadata
        | ConstantMetadata
        | TypeAliasMetadata
        | ControlFlowMetadata
    )
    docstring: str | None = None
    signature: Signature | None = None
    embedding: list[float] | None = None
    source_code: str | None = None
    header: str | None = None
    scope: list[str] = []  # e.g. ["my_module", "MyClass"] for a method

    def to_accumulated_item(self) -> "AccumulatedItem":
        """Convert to AccumulatedItem for retrieval context accumulation."""
        from backend.src.domain.schemas.retrieval import AccumulatedItem

        content_parts = []
        if self.signature:
            content_parts.append(self._signature_to_string())
        if self.docstring:
            content_parts.append(self.docstring)
        if self.source_code:
            lines = self.source_code.split("\n")
            content_parts.append("\n".join(lines[:20]))

        content = "\n\n".join(content_parts) if content_parts else self.name

        return AccumulatedItem(
            file_id=self.file_id,
            path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            content=content,
            source="graph",
            node_type=self.node_type.value,
        )

    def _signature_to_string(self) -> str:
        """Convert signature to string representation."""
        if not self.signature:
            return self.name

        sig = self.signature
        decorators = ""
        if sig.decorators:
            decorators = "".join(f"{d}\n" for d in sig.decorators)

        async_kw = "async " if sig.is_async else ""

        params = []
        if sig.parameters:
            for p in sig.parameters.params:
                param_str = p.name
                if p.type_annotation:
                    param_str += f": {p.type_annotation}"
                if p.default_value:
                    param_str += f" = {p.default_value}"
                params.append(param_str)

            if sig.parameters.accepts_args:
                params.append("*args")
            if sig.parameters.accepts_kwargs:
                params.append("**kwargs")
        params_str = ", ".join(params)

        receiver = ""
        if sig.receiver:
            receiver = f"({sig.receiver}) "

        return_str = ""
        if sig.return_type:
            return_str = f" -> {sig.return_type}"

        return f"{decorators}{async_kw}def {receiver}{self.name}({params_str}){return_str}:"


class GraphResult(DomainStruct):
    """complete result from code graph extraction."""

    file_id: str
    file_path: str
    language: str
    nodes: list[CodeNode]
    call_edges: list[CallEdge] = []
    inheritance_edges: list[InheritanceEdge] = []
    method_edges: list[MethodEdge] = []
    imports: list["ImportNode"] = []


class GraphQueryResult(DomainStruct):
    """result from a graph query (semantic or structural)."""

    nodes: list[CodeNode]
    paths: list[list[CodeNode]] = []
    total_count: int = 0
    query_type: str = "semantic"


# =============================================================================
# Intermediate parsing types (used during tree-sitter parsing)
# =============================================================================


class Point(DomainStruct):
    row: int
    column: int


class Range(DomainStruct):
    start_point: Point
    end_point: Point
    start_byte: int = 0
    end_byte: int = 0

    @classmethod
    def from_node(cls, node: "ts.Node") -> "Range":
        """create a Range from a tree-sitter Node."""
        sp = Point(row=node.start_point.row, column=node.start_point.column)
        ep = Point(row=node.end_point.row, column=node.end_point.column)
        return cls(
            start_point=sp,
            end_point=ep,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )


class Comment(DomainStruct):
    text: str
    row: int
    is_inline: bool


class Docstring(DomainStruct):
    content: str
    path: str


class Parameters(DomainStruct):
    params: list["Parameter"] = []
    accepts_kwargs: bool = False
    accepts_args: bool = False


class Parameter(DomainStruct):
    name: str
    type_annotation: str | None = None
    default_value: str | None = None


class Block(DomainStruct):
    statements: list["Node | Any"] = []


class Node(DomainStruct):
    """intermediate parse tree node — used during parsing, not graph output."""

    identifier: str
    text: str | None
    path: str
    range: Range
    type: str = ""
    parent_id: str | None = None
    children: list[Any] = []
    comments: list[Comment] = []
    file_id: str = ""


class ImportNode(Node):
    """intermediate import node for parsing."""

    statement: str = ""
    from_dotted_names: list[str] = []
    import_dotted_names: list[str] = []
    target_file_id: str | None = None
    target_file_path: str | None = None
    source_file_id: str = ""
    line_number: int | None = None
    aliased: str | None = None
    type: str = IMPORT.DEFAULT
