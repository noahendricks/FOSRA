from datetime import datetime
from typing import Any

from msgspec import field

from FOSRABack.src.domain.enums import (
    DocumentType,
    FileType,
    OriginType,
    StorageBackendType,
)
from FOSRABack.src.storage.models import utc_now
from FOSRABack.src.storage.utils.converters import DomainStruct


# =============================================================================
# Enums
# =============================================================================


# =============================================================================
# Configuration Dataclasses
# =============================================================================
#


class StorageConfig(DomainStruct):
    """Configuration for storage backend operations."""

    timeout_seconds: int = 30
    max_retries: int = 3
    chunk_size: int = 8192

    backend_options: dict[str, Any] = field(default_factory=dict)
    preferred_backend_type: StorageBackendType = StorageBackendType.FILESYSTEM


# =============================================================================
# Data Transfer Objects
# =============================================================================
class FileMetadata(DomainStruct):
    """Metadata about a file without content."""

    metadata_hash: str | None = None
    file_type: FileType = FileType.UNKNOWN
    file_name: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    origin_type: OriginType | None = None
    size: int = 0
    times_accessed: int | None = None
    last_accessed: datetime = datetime.now()


class FileContent(DomainStruct):
    """Content retrieved from a file."""

    file_path: str
    file_hash: str
    file_name: str
    content: bytes | str
    metadata: FileMetadata = field(default_factory=FileMetadata)


class File(DomainStruct):
    origin_path: str
    uploaded_at: datetime = field(default_factory=utc_now)
    name: str = ""
    hash: str | None = None


class FileProcessed(File):
    source_content: str | None = None
