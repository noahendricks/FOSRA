from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from qdrant_client.models import SparseVector

import msgspec
from backend.src.services.processing.utils.loader import code_mimes, text_mimes
from backend.src.storage.utils.converters import DomainStruct


class MDNFile(DomainStruct):
    type: str
    size: int
    name: str
    media_type: str
    # base64-encoded content when transmitted over JSON
    bytes: str | None = None
    url: str | None = None
    webkit_relative_path: str | None = None


# ─── Document metadata for various source types ─────────────────────────────────


class BaseDocumentMetadata(DomainStruct):
    source: str | None = None
    mime_type: str | None = None


class PDFMetadata(BaseDocumentMetadata, kw_only=True):
    content_type: Literal["pdf"] = "pdf"
    producer: str | None = None
    creator: str | None = None
    creationdate: datetime | str | None = None
    moddate: datetime | str | None = None
    modDate: str | None = None
    creationDate: str | None = None
    total_pages: int
    format: str | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    trapped: str | None = None
    file_path: str | None = None
    page: int | None = None

    def __post_init__(self) -> None:
        for field in ("creationdate", "moddate"):
            raw = getattr(self, field, None)
            if isinstance(raw, str) and raw.startswith("D:"):
                try:
                    clean = raw[2:].split("+")[0].split("-")[0].replace("'", "")
                    parsed = datetime.strptime(clean[:14], "%Y%m%d%H%M%S")
                    object.__setattr__(self, field, parsed)
                except ValueError:
                    pass


class TextMetadata(BaseDocumentMetadata):
    content_type: Literal["text", "html", "markdown", "txt", "csv"] = "text"


class CodeMetadata(BaseDocumentMetadata, kw_only=True):
    content_type: (
        Literal[
            "python",
            "cpp",
            "csharp",
            "cobol",
            "elixir",
            "go",
            "java",
            "js",
            "kotlin",
            "lua",
            "perl",
            "python",
            "ruby",
            "rust",
            "scala",
            "sql",
            "typescript",
        ]
        | None
    ) = None


class FunctionsClassesMetadata(CodeMetadata, kw_only=True):
    class_name: str | None = None
    function_name: str | None = None
    decorators: list[str] = []
    is_async: bool = False
    is_class_method: bool = False
    is_static_method: bool = False

    def __post_init__(self) -> None:
        if self.content_type is None:
            object.__setattr__(self, "content_type", "functions_classes")


class SimplifiedCodeMetadata(CodeMetadata, kw_only=True):
    original_length: Annotated[int | None, msgspec.Meta(ge=0)] = None
    simplified_ratio: Annotated[float | None, msgspec.Meta(ge=0.0, le=1.0)] = None

    def __post_init__(self) -> None:
        if self.content_type is None:
            object.__setattr__(self, "content_type", "simplified_code")


class ImportsMetadata(CodeMetadata, kw_only=True):
    is_third_party: bool = False
    is_stdlib: bool = False
    import_names: list[str] = []

    def __post_init__(self) -> None:
        if self.content_type is None:
            object.__setattr__(self, "content_type", "imports")


CodeMetadataUnion = FunctionsClassesMetadata | SimplifiedCodeMetadata | ImportsMetadata


class DocMetadata(DomainStruct, kw_only=True):
    source: str
    mime_type: str
    doc_id: str
    doc_title: str
    path: str | None = None
    language: str | None = None
    repo: str | None = None
    source_type: str = "doc"
    checksum: str | None = None
    sections: list[Section] = []
    section_heading: str | None = None  # active section heading during chunking


class Doc(DomainStruct, kw_only=True):
    id: str
    page_content: str
    metadata: DocMetadata

    @property
    def is_pdf(self) -> bool:
        return self.metadata.mime_type == "application/pdf"

    @property
    def is_code(self) -> bool:
        return self.metadata.mime_type in code_mimes.values()

    @property
    def is_text(self) -> bool:
        return self.metadata.mime_type in text_mimes.values()


class HierarchicalChunk(DomainStruct, kw_only=True):
    """a chunk in the hierarchical tree produced by hichunk."""

    text: str
    token_count: int
    level: int  # 1 = coarsest section, 2 = subsection, …
    start_char: int = 0
    end_char: int = 0
    children: list["HierarchicalChunk"] = []
    parent: "HierarchicalChunk | None" = None  # forward reference — OK in msgspec
    metadata: DocMetadata

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        snippet = self.text[:60].replace("\n", " ")
        return f"HierarchicalChunk(level={self.level}, tokens={self.token_count}, text='{snippet}…')"


class ChunkMetadata(DomainStruct, kw_only=True):
    chunk_id: str | None = None
    doc_id: str | None = None
    doc_title: str | None = None
    page_number: int | None = None
    token_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    dense_embedding: list[float] = []
    sparse_embedding: Any | SparseVector = None
    parent: Any = None  # typed as Any to skip msgspec structural validation
    section_heading: str | None = None
    element_ids: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Custom serialization — avoids recursive serialization of parent HierarchicalChunk (circular refs)."""
        d = {}
        for f in self.__struct_fields__:
            if f == "parent":
                continue
            v = getattr(self, f)
            d[f] = v
        if self.parent is not None:
            d["parent_text"] = getattr(self.parent, "text", None)
            d["parent_token_count"] = getattr(self.parent, "token_count", 0)
            d["parent_start_char"] = getattr(self.parent, "start_char", None)
            d["parent_end_char"] = getattr(self.parent, "end_char", None)
            d["parent_level"] = getattr(self.parent, "level", None)
        else:
            d["parent_text"] = None
            d["parent_token_count"] = 0
            d["parent_start_char"] = None
            d["parent_end_char"] = None
            d["parent_level"] = None
        return d


class Chunk(DomainStruct, kw_only=True):
    text: str
    metadata: ChunkMetadata


class ElementPosition(DomainStruct, kw_only=True):
    """Positional metadata for a single kreuzberg element."""

    page_number: int
    element_index: int
    element_id: str
    additional: dict[str, Any] | None = None


class Section(DomainStruct, kw_only=True):
    """A logical section of elements grouped by heading boundary."""

    elements: list[dict[str, Any]]  # kreuzberg element dicts
    section_text: str | None = None  # docling: combined text of all items in section
    heading: str | None = None  # section heading text (full path for docling)
    heading_path: list[str] | None = None  # docling: full heading hierarchy
    start_page: int | None = None
    end_page: int | None = None
    element_ids: list[str] = []
    section_index: int = 0
