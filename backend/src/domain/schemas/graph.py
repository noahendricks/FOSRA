from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.v1.utils import to_camel

from backend.src.domain.enums import GraphNodeType
from backend.src.storage.utils.converters import DomainStruct


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )
    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class Signature(_BaseModelFlex):
    """function signature with parameters and return type."""

    parameters: list[Parameter]
    return_type: str | None = None
    is_async: bool = False
    is_method: bool = False
    receiver: str | None = None
    decorators: list[str] = []


class Parameter(_BaseModelFlex):
    """a single function parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False
    is_keyword: bool = False


class CallEdge(_BaseModelFlex):
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


class InheritanceEdge(_BaseModelFlex):
    """represents a class inheritance/implementation relationship."""

    child_name: str
    child_qualified: str
    child_file_id: str
    parent_name: str
    parent_qualified: str | None = None
    parent_file_id: str | None = None
    inheritance_type: str = "extends"
    is_cross_file: bool = False


class ResolvedImport(_BaseModelFlex):
    """an import statement resolved to its target file."""

    import_statement: str
    imported_names: list[str]
    source_file_id: str
    target_file_id: str | None = None
    target_file_path: str | None = None
    line_number: int
    is_stdlib: bool = False
    is_third_party: bool = False


class CodeNode(_BaseModelFlex):
    """a node in the code graph (file, module, class, function, method)."""

    node_type: GraphNodeType
    name: str
    qualified_name: str
    file_id: str
    file_path: str
    line_start: int
    line_end: int
    signature: Signature | None = None
    docstring: str | None = None
    embedding: list[float] | None = None
    source_code: str | None = None
    metadata: dict[str, Any] = {}


class GraphResult(_BaseModelFlex):
    """complete result from code graph extraction."""

    file_id: str
    file_path: str
    language: str
    nodes: list[CodeNode]
    call_edges: list[CallEdge]
    inheritance_edges: list[InheritanceEdge]
    imports: list[ResolvedImport]

    @property
    def functions(self) -> list[CodeNode]:
        return [n for n in self.nodes if n.node_type == GraphNodeType.FUNCTION]

    @property
    def classes(self) -> list[CodeNode]:
        return [n for n in self.nodes if n.node_type == GraphNodeType.CLASS]

    @property
    def methods(self) -> list[CodeNode]:
        return [n for n in self.nodes if n.node_type == GraphNodeType.METHOD]


class GraphQueryResult(_BaseModelFlex):
    """result from a graph query (semantic or structural)."""

    nodes: list[CodeNode]
    paths: list[list[CodeNode]] = []
    total_count: int = 0
    query_type: str = "semantic"
