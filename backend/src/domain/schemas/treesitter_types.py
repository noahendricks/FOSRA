from __future__ import annotations

from enum import StrEnum
from typing import Any

import tree_sitter as ts

from backend.src.storage.utils.converters import DomainStruct


# Graph node types for flattened storage
class GraphNodeType(StrEnum):
    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    CONSTANT = "Constant"  # module-level assignments
    IMPORT = "Import"  # import statements as nodes
    TYPE_ALIAS = "TypeAlias"  # type: ignore


# Module
class MODULE(StrEnum):
    MODULE = "module"


# Imports
class IMPORT(StrEnum):
    DEFAULT = "import_statement"
    FROM = "import_from_statement"
    FUTURE = "future_import_statement"


# Class/Function
class DEF(StrEnum):
    CLASS = "class_definition"
    CLASS_DECORATED = "decorated_definition"
    FUNCTION = "function_definition"


# Module-level definitions
class MODULE_DEF(StrEnum):
    CONSTANT = "assignment"  # module-level assignments
    TYPE_ALIAS = "type_alias_statement"


MODULE_DEF_TYPES: set[str] = {MODULE_DEF.CONSTANT, MODULE_DEF.TYPE_ALIAS}


# Compound statements
class COMPOUND(StrEnum):
    IF = "if_statement"
    FOR = "for_statement"
    WHILE = "while_statement"
    TRY = "try_statement"
    WITH = "with_statement"
    MATCH = "match_statement"


# Simple statements
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


class COMMENT(StrEnum):
    COMMENT = "comment"


# Expressions (callable/literal)
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
# Type sets for membership checks
# =============================================================================

IMPORT_TYPES: set[str] = {IMPORT.DEFAULT, IMPORT.FROM, IMPORT.FUTURE}
CLASS_TYPES: set[str] = {DEF.CLASS}
FUNCTION_TYPES: set[str] = {DEF.FUNCTION}
DEFINITION_TYPES: set[str] = CLASS_TYPES | FUNCTION_TYPES

COMPOUND_TYPES: set[str] = {
    COMPOUND.IF,
    COMPOUND.FOR,
    COMPOUND.WHILE,
    COMPOUND.TRY,
    COMPOUND.WITH,
    COMPOUND.MATCH,
}
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
COMMENT_TYPES: set[str] = {COMMENT.COMMENT}
STATEMENT_TYPES: set[str] = IMPORT_TYPES | SIMPLE_TYPES | COMMENT_TYPES

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
    LITERAL.STRING,
    LITERAL.INTEGER,
    LITERAL.FLOAT,
    LITERAL.TRUE,
    LITERAL.FALSE,
    LITERAL.NONE,
}
STRUCTURAL_TYPES: set[str] = {
    STRUCT.IDENTIFIER,
    STRUCT.TYPE_PARAM,
    STRUCT.DOTTED_NAME,
    STRUCT.KW_IDENTIFIER,
    STRUCT.SLICE,
    STRUCT.ELLIPSIS,
    STRUCT.PATTERN_LIST,
    STRUCT.GENERIC_TYPE,
    STRUCT.UNION_TYPE,
}
PATTERN_TYPES: set[str] = {
    PATTERN.CLASS,
    PATTERN.DICT,
    PATTERN.LIST,
    PATTERN.TUPLE,
}


def is_comment(node_type: str) -> bool:
    return node_type in COMMENT_TYPES


ALL_TYPES: set[str] = (
    {MODULE.MODULE}
    | IMPORT_TYPES
    | CLASS_TYPES
    | FUNCTION_TYPES
    | COMPOUND_TYPES
    | SIMPLE_TYPES
    | EXPRESSION_TYPES
    | LITERAL_TYPES
    | STRUCTURAL_TYPES
    | PATTERN_TYPES
    | COMMENT_TYPES
)


# =============================================================================
# Top-level namespace for match statements
# =============================================================================


class T:
    """Namespace for tree-sitter type constants. Usage: T.IMPORT, T.CLASS, etc."""

    IMPORT = IMPORT_TYPES
    DEF = [CLASS_TYPES, FUNCTION_TYPES]
    CLASS = CLASS_TYPES
    FUNCTION = FUNCTION_TYPES
    COMPOUND = COMPOUND_TYPES
    SIMPLE = SIMPLE_TYPES
    EXPRESSION = EXPRESSION_TYPES
    LITERAL = LITERAL_TYPES
    STRUCTURAL = STRUCTURAL_TYPES
    PATTERN = PATTERN_TYPES
    MODULE = {MODULE.MODULE}
    COMMENT = COMMENT_TYPES


def get_type(node_type: str) -> str | None:
    """get the type string if it's a valid tree-sitter type."""
    return node_type if node_type in ALL_TYPES else None


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


def is_literal(node_type: str) -> bool:
    return node_type in LITERAL_TYPES


def is_structural(node_type: str) -> bool:
    return node_type in STRUCTURAL_TYPES


def is_pattern(node_type: str) -> bool:
    return node_type in PATTERN_TYPES


def is_module(node_type: str) -> bool:
    return node_type == MODULE.MODULE


def is_decorated(node_type: str) -> bool:
    return node_type == DEF.CLASS_DECORATED


def is_await(node_type: str) -> bool:
    return node_type == EXPR.AWAIT


def is_dotted_name(node_type: str) -> bool:
    return node_type == STRUCT.DOTTED_NAME


__all__ = [
    # Graph node types
    "GraphNodeType",
    # Hierarchical enums
    "MODULE",
    "IMPORT",
    "DEF",
    "MODULE_DEF",
    "COMPOUND",
    "SIMPLE",
    "EXPR",
    "LITERAL",
    "STRUCT",
    "PATTERN",
    "COMMENT",
    # Namespace
    "T",
    # Sets
    "IMPORT_TYPES",
    "CLASS_TYPES",
    "FUNCTION_TYPES",
    "DEFINITION_TYPES",
    "MODULE_DEF_TYPES",
    "COMPOUND_TYPES",
    "SIMPLE_TYPES",
    "STATEMENT_TYPES",
    "EXPRESSION_TYPES",
    "LITERAL_TYPES",
    "STRUCTURAL_TYPES",
    "PATTERN_TYPES",
    "COMMENT_TYPES",
    "ALL_TYPES",
    # Helpers
    "get_type",
    "is_import",
    "is_class",
    "is_function",
    "is_definition",
    "is_compound",
    "is_simple",
    "is_statement",
    "is_comment",
    "is_expression",
    "is_literal",
    "is_structural",
    "is_pattern",
    "is_module",
    "is_decorated",
    "is_await",
    "is_dotted_name",
]


# =============================================================================
# Domain structs for typed AST nodes
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
    def from_node(cls, node: "ts.Node") -> Range:
        """create a Range from a tree-sitter Node, extracting start/end points and byte offsets."""
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


class Node(DomainStruct):
    identifier: str
    text: str | None
    path: str
    range: Range
    type: str = ""
    parent_id: str | None = None
    children: list[Any] = []
    comments: list[Comment] = []


# treesitter_types.py | import_statement{annotations}
# treesitter_types.py | import_statement {from: pathlib.Path import: path }
# treesitter_types.py | future_import_statement {from: __future__ import: annotations }


class ImportNode(Node):
    from_dotted_names: list[str] = []
    import_dotted_names: list[str] = []
    aliased: str | None = None
    type: str = IMPORT.DEFAULT
    # TODO: Remove children from import node / only nodes with children have children field


class SimpleNode(Node):
    type: str = ""
    statement_type: str = ""


class ClassNode(Node):
    pass


class TypeAnnotation(DomainStruct):
    type_name: str
    generic_type: GenericType | None


class GenericType(DomainStruct):
    base_type: str
    type_parameters: list[str] | None


class Dictionary(DomainStruct):
    is_empty: bool
    entries: list[DictEntry] | None


class DictEntry(DomainStruct):
    key: str
    value: str


class AssignmentNode(SimpleNode):
    name: str = ""
    type_annotation: TypeAnnotation | None = None
    value: Any | None = None


class ExpressionStatement(SimpleNode):
    pass


class Block(DomainStruct):
    statements: list[Node | Any] = []  # SimpleNode, CompoundNode, or other typed nodes


class Parameters(DomainStruct):
    params: dict[str, str] = {}


class Parameter(DomainStruct):
    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False
    is_keyword: bool = False


class TypedParameter(DomainStruct):
    name: str
    type_annotation: str
    default_value: str | None = None


class DecoratorNode(DomainStruct):
    name: str = ""


class FunctionNode(SimpleNode):
    name: str = ""
    parameters: Parameters | None = None
    return_type: str | None = None
    body: Block | None = None
    is_async: bool = False
    decorators: list[str] = []
    receiver: str | None = None
    containing_class: str | None = None
    docstring: Docstring | None = None


class DecoratedDefinition(SimpleNode):
    decorator: DecoratorNode | None = None
    definition: FunctionNode | None = None


class ClassDefinition(Node):
    name: str = ""
    superclasses: list[str] | None = None
    body: Block | None = None
    docstring: Docstring | None = None
    decorators: list[str] = []
    methods: list[str] = []


class Call(DomainStruct):
    function: str = ""
    arguments: list[Any] = []
    is_method_call: bool = False


class AttributeAccess(DomainStruct):
    object: str = ""
    attribute: str = ""
    is_subscript: bool = False
    subscript_index: Any | None = None


class BinaryOperator(DomainStruct):
    operator: str = ""
    left: Any = None
    right: Any = None


class ComparisonOperator(DomainStruct):
    operators: list[str] = []
    operands: list[Any] = []


class BooleanOperator(DomainStruct):
    operator: str = ""
    left: Any = None
    right: Any = None


class ConditionalExpression(DomainStruct):
    condition: Any = None
    true_branch: Any = None
    false_branch: Any = None


class Lambda(DomainStruct):
    parameters: str = ""
    body: Any = None


class NamedExpression(DomainStruct):
    name: str = ""
    value: Any = None


class ListLiteral(DomainStruct):
    elements: list[Any] = []
    is_empty: bool = True


class TupleLiteral(DomainStruct):
    elements: list[Any] = []
    is_empty: bool = True


class SetLiteral(DomainStruct):
    elements: list[Any] = []
    is_empty: bool = True


class ForStatement(Node):
    target: str = ""
    iterable: Any = None
    body: Block | None = None
    else_body: Block | None = None
    is_async: bool = False


class WhileStatement(Node):
    condition: Any = None
    body: Block | None = None
    else_body: Block | None = None


class TryStatement(Node):
    body: Block | None = None
    handlers: list[ExceptHandler] = []
    else_body: Block | None = None
    finally_body: Block | None = None


class ExceptHandler(DomainStruct):
    exception_type: str | None = None
    alias: str | None = None
    body: Block | None = None


class WithItem(DomainStruct):
    value: str = ""
    alias: str | None = None


class WithStatement(Node):
    items: list[WithItem] = []
    body: Block | None = None
    is_async: bool = False


class MatchStatement(Node):
    subject: Any = None
    cases: list[CaseClause] = []


class CaseClause(DomainStruct):
    pattern: Any = None
    guard: Any | None = None
    body: Block | None = None


class AugmentedAssignment(SimpleNode):
    name: str = ""
    operator: str = ""
    value: Any = None


class RaiseStatement(SimpleNode):
    exception: Any | None = None
    cause: Any | None = None


class AssertStatement(SimpleNode):
    condition: Any = None
    message: Any | None = None


class DeleteStatement(SimpleNode):
    target: Any = None


class GlobalStatement(SimpleNode):
    names: list[str] = []


class NonlocalStatement(SimpleNode):
    names: list[str] = []


class YieldExpression(DomainStruct):
    value: Any | None = None
    is_yield_from: bool = False


class BreakStatement(SimpleNode):
    pass


class ContinueStatement(SimpleNode):
    pass


class PassStatement(SimpleNode):
    pass


class TypeAliasStatement(SimpleNode):
    name: str = ""
    value: Any = None


class ReturnStatement(SimpleNode):
    value: Any | None = None


class IfStatement(Node):
    condition: Any = None
    body: Block | None = None
    elif_body: Block | None = None
    else_body: Block | None = None


class CompoundNode(Node):
    """Compound statement node — text = header line only, body = children."""

    header: str = ""
    body_type: str = (
        ""  # "if", "for", "while", "with", "try", "elif", "else", "except", "finally"
    )


class Identifier(DomainStruct):
    name: str = ""
