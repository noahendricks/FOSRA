from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from chonkie.types import Chunk as ChonkieChunk
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel
from qdrant_client.models import SparseVector

from backend.src.api.schemas.base import _BaseModelFlex
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


class BaseDocumentMetadata(BaseModel):
    source: str | None = None
    mime_type: str | None = None


class PDFMetadata(BaseDocumentMetadata):
    content_type: Literal["pdf"] = "pdf"

    producer: str | None = None
    creator: str | None = None
    creationdate: datetime | str | None = None
    moddate: datetime | str | None = None
    modDate: str | None = Field(None, alias="modDate")
    creationDate: str | None = Field(None, alias="creationDate")
    total_pages: int = Field(..., gt=0)
    format: str | None = None  # e.g., "PDF 1.7"
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    trapped: str | None = None
    file_path: str | None = None

    page: int | None = Field(None, ge=0)

    @field_validator("creationdate", "moddate", mode="before")
    @classmethod

    # handle pdf date strings like 'D:20200331174925+02'00''
    def parse_pdf_date(cls, v):
        if isinstance(v, str) and v.startswith("D:"):
            # strip the d: prefix and try to parse
            try:
                # extract just the datetime part, ignore timezone offset for simplicity
                clean = v[2:].split("+")[0].split("-")[0].replace("'", "")
                return datetime.strptime(clean[:14], "%Y%m%d%H%M%S")
            except ValueError:
                return v  # return as string if parsing fails
        return v


class TextMetadata(BaseDocumentMetadata):
    content_type: Literal["text", "html", "markdown", "txt", "csv"] = "text"


class CodeMetadata(BaseDocumentMetadata):
    language: Literal[
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


# extracted functions and classes from code.
class FunctionsClassesMetadata(CodeMetadata):
    content_type: Literal["functions_classes"] = "functions_classes"
    class_name: str | None = None
    function_name: str | None = None
    decorators: list[str] = Field(default_factory=list)
    is_async: bool = False
    is_class_method: bool = False
    is_static_method: bool = False


# simplified/minified code representation.
class SimplifiedCodeMetadata(CodeMetadata):
    content_type: Literal["simplified_code"] = "simplified_code"
    original_length: int | None = Field(None, ge=0)
    simplified_ratio: float | None = Field(None, ge=0.0, le=1.0)


# extracted import statements
class ImportsMetadata(CodeMetadata):
    content_type: Literal["imports"] = "imports"
    is_third_party: bool = False
    is_stdlib: bool = False
    import_names: list[str] = Field(default_factory=list)


CodeMetadataUnion = FunctionsClassesMetadata | SimplifiedCodeMetadata | ImportsMetadata


# main doc with typed metadata - used everywhere Document(LC) would be


class DocMetadata(_BaseModelFlex):
    source: str
    mime_type: str
    doc_id: str
    doc_title: str
    path: str | None = None
    language: str | None = None
    repo: str | None = None
    source_type: str = "doc"
    checksum: str | None = None


class Doc(BaseModel):
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


class HierarchicalChunk(_BaseModelFlex):
    """a chunk in the hierarchical tree produced by hichunk."""

    text: str
    token_count: int
    level: int  # 1 = coarsest section, 2 = subsection, …
    start_char: int = 0
    end_char: int = 0
    children: list["HierarchicalChunk"] = Field(default_factory=list)
    parent: "HierarchicalChunk | None" = Field(default=None, validate_default=False)
    metadata: DocMetadata

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        snippet = self.text[:60].replace("\n", " ")
        return f"HierarchicalChunk(level={self.level}, tokens={self.token_count}, text='{snippet}…')"


class ChunkMetadata(_BaseModelFlex):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
        validate_default=False,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore

    chunk_id: str | None = None
    doc_id: str | None = None
    doc_title: str | None = None
    page_number: int | None = None
    token_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    dense_embedding: list[float] = []
    sparse_embedding: Any | SparseVector = None
    parent: HierarchicalChunk | None = Field(default=None, validate_default=False)


class Chunk(_BaseModelFlex):
    text: str
    metadata: ChunkMetadata

    @classmethod
    def from_chonkie(cls, chunk: ChonkieChunk) -> Self:

        _meta = ChunkMetadata(
            chunk_id=chunk.id,
            start_char=chunk.start_index,
            end_char=chunk.end_index,
            token_count=chunk.token_count,
        )

        return cls(text=chunk.text, metadata=_meta)
