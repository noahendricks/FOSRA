from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from qdrant_client.models import SparseVector

from backend.src.services.processing.utils.parse_utils import code_mimes, text_mimes
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


class RawBlob(DomainStruct, kw_only=True, frozen=True):
    """raw data abstraction for document loading and file processing."""

    data: bytes | str | None = None
    """raw data associated with the `Blob`."""

    mimetype: str | None = None
    """mime type, not to be confused with a file extension."""

    encoding: str = "utf-8"
    """encoding to use if decoding the bytes into a string."""

    path: str | None = None
    """location where the original content was found."""

    @classmethod
    def from_data(
        cls,
        data: bytes | str,
        mimetype: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> "RawBlob":
        return cls(data=data, mimetype=mimetype, path=path, **kwargs)

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "RawBlob":
        p = Path(path)
        with open(p, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
            return cls(data=text, mimetype="text/plain", path=path, **kwargs)
        except UnicodeDecodeError:
            return cls(data=raw, mimetype=None, path=path, **kwargs)

    def as_string(self) -> str:
        """Read the blob as a string."""
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        raise ValueError("No data available")

    def as_bytes(self) -> bytes:
        """Read the blob as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, str):
            return self.data.encode(self.encoding)
        raise ValueError("No data available")

    def as_bytes_io(self):
        """Read the blob as a byte stream."""
        from io import BytesIO

        return BytesIO(self.as_bytes())


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
        return self.metadata.mime_type in code_mimes

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


class SectionMetadata(DomainStruct, kw_only=True):
    section_id: str | None = None
    parent_id: str | None = None  # section_id of parent; None = root
    doc_id: str | None = None
    doc_title: str | None = None
    page_number: int | None = None
    token_count: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    dense_embedding: list[float] = []
    sparse_embedding: Any | SparseVector = None
    section_heading: str | None = None
    heading_level: int | None = None  # depth in heading hierarchy (1 = outermost)
    heading_path: list[str] | None = None  # full path from root heading to this heading
    # code-specific metadata
    source_file: str | None = None
    code_definition_type: (
        Literal["class", "function", "method", "async_function"] | None
    ) = None
    is_async: bool = False
    is_method: bool = False
    decorators: list[str] | None = None
    docstring: str | None = None
    parameters: list[str] | None = None
    return_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for f in self.__struct_fields__:
            d[f] = getattr(self, f)
        return d


# backward-compat alias — remove once all callers are updated
ChunkMetadata = SectionMetadata


class Subsection(DomainStruct, kw_only=True):
    """a unit of content emitted by ingestion — either a section or a split child."""

    text: str
    metadata: SectionMetadata


# backward-compat alias
Chunk = Subsection


class Section(DomainStruct, kw_only=True):
    """a logical section from a document, carrying its heading hierarchy and child sections."""

    section_text: str | None = None
    heading: str | None = None
    heading_path: list[str] | None = (
        None  # full breadcrumb from root heading to this one
    )
    start_page: int | None = None
    end_page: int | None = None
    section_index: int = 0
    section_id: str | None = (
        None  # "{doc_id}:{section_index}", assigned at tree-build time
    )
    parent_id: str | None = None  # section_id of parent; None = root section
    children: list["Section"] = []  # populated during ingestion, not stored in qdrant
