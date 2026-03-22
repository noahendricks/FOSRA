from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic.v1.utils import to_camel

from backend.src.domain.enums import GraphNodeType
from backend.src.storage.utils.converters import DomainStruct

if TYPE_CHECKING:
    from backend.src.domain.schemas.retrieval import AccumulatedItem


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
        for p in sig.parameters:
            param_str = p.name
            if p.type_annotation:
                param_str += f": {p.type_annotation}"
            if p.default_value:
                param_str += f" = {p.default_value}"
            if p.is_variadic:
                param_str = f"*{param_str}"
            elif p.is_keyword:
                param_str = f"**{param_str}"
            params.append(param_str)

        params_str = ", ".join(params)

        receiver = ""
        if sig.receiver:
            receiver = f"({sig.receiver}) "

        return_str = ""
        if sig.return_type:
            return_str = f" -> {sig.return_type}"

        return f"{decorators}{async_kw}def {receiver}{self.name}({params_str}){return_str}:"


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
