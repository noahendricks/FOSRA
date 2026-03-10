from __future__ import annotations

import builtins
from datetime import datetime
from typing import Any, Literal, Self

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.v1.utils import to_camel
from qdrant_client.models import SparseVector

from backend.src.services.processing.hi_chunk import HierarchicalChunk
from backend.src.storage.models import ulid_factory
from backend.src.storage.utils.converters import DomainStruct


class _BaseModelFlex(BaseModel):
    _FLEXIBLE_CONFIG = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    model_config: ConfigDict = _FLEXIBLE_CONFIG  # pyright: ignore


class MDNFile(DomainStruct):
    type: str
    size: int
    name: str
    bytes: builtins.bytes | str | None
    media_type: str
    url: str | None = None
    webkit_relative_path: str | None = None


class BaseDocumentMetadata(BaseModel):
    source: str | None = None


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
class Doc(BaseModel):
    id: str
    type: Literal["Document"] = "Document"
    page_content: str
    metadata: (
        PDFMetadata
        | TextMetadata
        | FunctionsClassesMetadata
        | SimplifiedCodeMetadata
        | ImportsMetadata
    )

    @classmethod
    def from_lc(cls, doc: Document) -> Self:
        meta_dict = doc.metadata or {}
        content_type = meta_dict.get("content_type", "text")

        # route to correct metadata model
        metadata_map = {
            "pdf": PDFMetadata,
            "text": TextMetadata,
            "html": TextMetadata,
            "markdown": TextMetadata,
            "functions_classes": FunctionsClassesMetadata,
            "simplified_code": SimplifiedCodeMetadata,
            "imports": ImportsMetadata,
        }

        model_class = metadata_map.get(content_type, TextMetadata)

        return cls(
            page_content=doc.page_content,
            metadata=model_class.model_validate(meta_dict),
            id=str(ulid_factory()),
        )

    def to_lc(self) -> Document:
        return Document(
            page_content=self.page_content,
            metadata=self.metadata.model_dump(by_alias=True, exclude_none=True),
            id=self.id,
        )

    @property
    def is_pdf(self) -> bool:
        return isinstance(self.metadata, PDFMetadata)

    @property
    def is_code(self) -> bool:
        return isinstance(
            self.metadata,
            (FunctionsClassesMetadata, SimplifiedCodeMetadata, ImportsMetadata),
        )

    @property
    def is_text(self) -> bool:
        return isinstance(self.metadata, TextMetadata)


class ChunkMetadata(_BaseModelFlex):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int | None
    token_count: int | None
    start_index: int | None
    end_index: int | None
    start_char: str | None
    end_char: str | None
    parent: HierarchicalChunk | None = None
    dense_embedding: list[float] = []
    sparse_embedding: Any | SparseVector = None
    late_embedding: list[float] = []


class Chunk(_BaseModelFlex):
    text: str
    metadata: ChunkMetadata
