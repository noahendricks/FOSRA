"""tree-sitter type enums, constants, and intermediate parsing types.

This module is the source of truth for tree-sitter type constants and intermediate
parsing types. The type constants (enums, sets, helper functions) are re-exported
from graph_types.py for consolidation, but the intermediate parsing types remain here
since they're used directly by parsing_funcs.py during the parsing pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import tree_sitter as ts

from backend.src.storage.utils.converters import DomainStruct

# Re-export type constants from graph_types.py for consolidation
from backend.src.domain.schemas.graph_types import (
    CLASS_TYPES,
    COMPOUND,
    COMPOUND_TYPES,
    COMMENT,
    COMMENT_TYPES,
    DEF,
    DEFINITION_TYPES,
    EXPR,
    EXPRESSION_TYPES,
    FUNCTION_TYPES,
    GraphNodeType,
    IMPORT,
    IMPORT_TYPES,
    LITERAL,
    LITERAL_TYPES,
    MODULE,
    MODULE_DEF,
    MODULE_DEF_TYPES,
    PATTERN,
    PATTERN_TYPES,
    SIMPLE,
    SIMPLE_TYPES,
    STATEMENT_TYPES,
    STRUCT,
    STRUCTURAL_TYPES,
    is_class,
    is_compound,
    is_decorated,
    is_definition,
    is_dotted_name,
    is_expression,
    is_function,
    is_import,
    is_simple,
    is_statement,
)


# =============================================================================
# Domain structs for intermediate parsing types (used during parsing, not graph output)
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


class SimpleNode(Node):
    type: str = ""
    statement_type: str = ""


class TypeAnnotation(DomainStruct):
    type_name: str
    generic_type: GenericType | None = None


class GenericType(DomainStruct):
    base_type: str
    type_parameters: list[str] | None = None


class Dictionary(DomainStruct):
    is_empty: bool
    entries: list[DictEntry] | None = None


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
    statements: list[Node | Any] = []


class Parameters(DomainStruct):
    params: list[Parameter] = []
    accepts_kwargs: bool = False
    accepts_args: bool = False


class Parameter(DomainStruct):
    name: str
    type_annotation: str | None = None
    default_value: str | None = None


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
    header: str | None = None


class DecoratedDefinition(SimpleNode):
    decorator: DecoratorNode | None = None
    definition: FunctionNode | None = None


class ClassNode(Node):
    name: str = ""
    superclasses: list[str] | None = None
    body: Block | None = None
    docstring: Docstring | None = None
    decorators: list[str] = []
    methods: list[str] = []
    header: str | None = None


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
    header: str = ""
    body_type: str = (
        ""  # "if", "for", "while", "with", "try", "elif", "else", "except", "finally"
    )


class Identifier(DomainStruct):
    name: str = ""


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
    # Helpers
    "is_import",
    "is_class",
    "is_function",
    "is_definition",
    "is_compound",
    "is_simple",
    "is_statement",
    "is_expression",
    "is_decorated",
    "is_dotted_name",
    # Intermediate parsing types
    "Point",
    "Range",
    "Comment",
    "Docstring",
    "Node",
    "ImportNode",
    "SimpleNode",
    "TypeAnnotation",
    "GenericType",
    "Dictionary",
    "DictEntry",
    "AssignmentNode",
    "ExpressionStatement",
    "Block",
    "Parameters",
    "Parameter",
    "TypedParameter",
    "DecoratorNode",
    "FunctionNode",
    "DecoratedDefinition",
    "ClassNode",
    "Call",
    "AttributeAccess",
    "BinaryOperator",
    "ComparisonOperator",
    "BooleanOperator",
    "ConditionalExpression",
    "Lambda",
    "NamedExpression",
    "ListLiteral",
    "TupleLiteral",
    "SetLiteral",
    "ForStatement",
    "WhileStatement",
    "TryStatement",
    "ExceptHandler",
    "WithItem",
    "WithStatement",
    "MatchStatement",
    "CaseClause",
    "AugmentedAssignment",
    "RaiseStatement",
    "AssertStatement",
    "DeleteStatement",
    "GlobalStatement",
    "NonlocalStatement",
    "YieldExpression",
    "BreakStatement",
    "ContinueStatement",
    "PassStatement",
    "TypeAliasStatement",
    "ReturnStatement",
    "IfStatement",
    "CompoundNode",
    "Identifier",
]
