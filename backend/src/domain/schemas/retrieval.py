"""Domain schemas for the evolved retrieval pipeline.

Implements the query expansion + agentic retrieval loop pattern from FULL.md:
    expand_query → initial_retrieve → agentic_loop → rerank → END

Key types:
    - ChecklistItem: A single coverage question with answered status
    - QueryExpansion: rewritten query + checklist
    - RetrievalQuery: Targeted retrieval command from subagent
    - SubagentResult: Subagent output (checklist update + retrieval queries)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RetrievalTarget(StrEnum):
    """Where to execute a retrieval query."""

    VECTOR = "vector"
    GRAPH = "graph"
    BOTH = "both"


class RetrievalFilters(BaseModel):
    """Filters for retrieval queries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_ids: list[str] | None = None
    node_type: str | None = None
    language: str | None = None


class ChecklistItem(BaseModel):
    """A single coverage question with answered status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    question: str
    answered: bool = False


class QueryExpansion(BaseModel):
    """Result of query expansion (Phase 1 of retrieval)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rewritten_query: str
    checklist: list[ChecklistItem]


class RetrievalQuery(BaseModel):
    """A targeted retrieval command emitted by the subagent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    target: RetrievalTarget = RetrievalTarget.VECTOR
    filters: RetrievalFilters | None = None


class SubagentResult(BaseModel):
    """Result of a single subagent iteration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checklist: list[ChecklistItem]
    all_answered: bool = False
    retrieval_queries: list[RetrievalQuery] = []


class AccumulatedItem(BaseModel):
    """A single item in the accumulated context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_id: str
    path: str
    line_start: int = 0
    line_end: int = 0
    content: str
    source: str = "vector"
    score: float = 0.0
    node_type: str | None = None
    qdrant_point_id: str | None = None


class AccumulatedContext(BaseModel):
    """Context accumulated across retrieval iterations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class GraphSearchResult(BaseModel):
    """Result from a graph search (semantic or structural)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
