from __future__ import annotations

from enum import StrEnum

from backend.src.storage.utils.converters import DomainStruct


class RetrievalTarget(StrEnum):
    """Where to execute a retrieval query."""

    VECTOR = "vector"
    GRAPH = "graph"
    BOTH = "both"


class RetrievalFilters(DomainStruct, kw_only=True, frozen=True):
    """Filters for retrieval queries."""

    file_ids: list[str] | None = None
    node_type: str | None = None
    language: str | None = None


class ChecklistItem(DomainStruct, kw_only=True, frozen=True):
    """A single coverage question with answered status."""

    id: str
    question: str
    answered: bool = False


class QueryExpansion(DomainStruct, kw_only=True, frozen=True):
    """Result of query expansion (Phase 1 of retrieval)."""

    rewritten_query: str
    checklist: list[ChecklistItem]


class RetrievalQuery(DomainStruct, kw_only=True, frozen=True):
    """A targeted retrieval command emitted by the subagent."""

    query: str
    target: RetrievalTarget = RetrievalTarget.VECTOR
    filters: RetrievalFilters | None = None


class SubagentResult(DomainStruct, kw_only=True, frozen=True):
    """Result of a single subagent iteration."""

    checklist: list[ChecklistItem]
    all_answered: bool = False
    retrieval_queries: list[RetrievalQuery] = []


class AccumulatedItem(DomainStruct, kw_only=True, frozen=True):
    """A single item in the accumulated context."""

    file_id: str
    path: str
    line_start: int = 0
    line_end: int = 0
    content: str
    source: str = "vector"
    score: float = 0.0
    node_type: str | None = None
    qdrant_point_id: str | None = None


class AccumulatedContext(DomainStruct, kw_only=True, frozen=True):
    """Context accumulated across retrieval iterations."""

    items: list[AccumulatedItem] = []

    def add_items(self, new_items: list[AccumulatedItem]) -> "AccumulatedContext":
        """Add new items, deduplicating by file_id + line_start."""
        existing_keys = {(i.file_id, i.line_start) for i in self.items}
        unique = [
            i for i in new_items if (i.file_id, i.line_start) not in existing_keys
        ]
        return AccumulatedContext(items=self.items + unique)

    def to_plain_text(self) -> str:
        """Concatenate all content for LLM consumption."""
        return "\n\n".join(i.content for i in self.items)

    def to_formatted_context(self) -> str:
        """Format as XML with citation metadata."""
        if not self.items:
            return ""

        parts = []
        for i, item in enumerate(self.items):
            parts.append(
                f"<source id='{i}' file_id='{item.file_id}' "
                f"path='{item.path}' lines='{item.line_start}-{item.line_end}'>\n"
                f"{item.content}\n"
                f"</source>"
            )

        return "<documents>\n" + "\n".join(parts) + "\n</documents>"


class GraphSearchResult(DomainStruct, kw_only=True, frozen=True):
    """Result from a graph search (semantic or structural)."""

    node_type: str
    name: str
    qualified_name: str
    file_id: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None = None
    signature: str | None = None
    score: float = 0.0

    def to_accumulated_item(self) -> AccumulatedItem:
        """Convert to an AccumulatedItem for context accumulation."""
        content_parts = []
        if self.signature:
            content_parts.append(self.signature)
        if self.docstring:
            content_parts.append(self.docstring)

        content = "\n\n".join(content_parts) if content_parts else self.name

        return AccumulatedItem(
            file_id=self.file_id,
            path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            content=content,
            source="graph",
            score=self.score,
            node_type=self.node_type,
        )
