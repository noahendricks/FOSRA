"""Tests for OriginService and storage backends.

Provides comprehensive test coverage for:
- BaseStorageBackend registry and factory methods
- FilesystemBackend operations
- S3Backend operations (mocked)
- HTTPBackend operations (mocked)
- OriginService unified interface
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from FOSRABack.src.retrieval.services.origin_service import (
    BaseStorageBackend,
    FilesystemBackend,
    HTTPBackend,
    OriginContent,
    OriginMetadata,
    OriginService,
    S3Backend,
    StorageBackendType,
    StorageConfig,
)
from FOSRABack.src.storage.models import OriginORM
from FOSRABack.src.storage.schemas import ConnectorType, OriginType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Hello, World!")

        sub_dir = Path(tmpdir) / "subdir"
        sub_dir.mkdir()
        sub_file = sub_dir / "nested.md"
        sub_file.write_text("# Nested content")

        yield tmpdir


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def filesystem_backend():
    """Create a filesystem backend instance."""
    return FilesystemBackend()


@pytest.fixture
def s3_backend():
    """Create an S3 backend instance."""
    return S3Backend()


@pytest.fixture
def http_backend():
    """Create an HTTP backend instance."""
    return HTTPBackend()


@pytest.fixture
def origin_service(mock_session):
    """Create an origin service instance."""
    return OriginService(session=mock_session)


# =============================================================================
# BaseStorageBackend Tests
# =============================================================================


class TestBaseStorageBackend:
    """Tests for BaseStorageBackend class methods."""

    def test_get_available_backends(self):
        """Test getting list of registered backends."""
        backends = BaseStorageBackend.get_available_backends()
        assert StorageBackendType.FILESYSTEM in backends
        assert StorageBackendType.S3 in backends
        assert StorageBackendType.HTTP in backends

    def test_get_backend_filesystem(self):
        """Test factory method for filesystem backend."""
        backend = BaseStorageBackend.get_backend(StorageBackendType.FILESYSTEM)
        assert backend is not None
        assert isinstance(backend, FilesystemBackend)

    def test_get_backend_s3(self):
        """Test factory method for S3 backend."""
        backend = BaseStorageBackend.get_backend(StorageBackendType.S3)
        assert backend is not None
        assert isinstance(backend, S3Backend)

    def test_get_backend_http(self):
        """Test factory method for HTTP backend."""
        backend = BaseStorageBackend.get_backend(StorageBackendType.HTTP)
        assert backend is not None
        assert isinstance(backend, HTTPBackend)

    def test_get_backend_unknown(self):
        """Test factory method with unknown backend type."""
        # Using a non-registered type
        backend = BaseStorageBackend.get_backend(StorageBackendType.GCS)
        assert backend is None

    def test_detect_backend_type_s3(self):
        """Test backend type detection for S3 paths."""
        assert (
            BaseStorageBackend.detect_backend_type("s3://bucket/key")
            == StorageBackendType.S3
        )
        assert (
            BaseStorageBackend.detect_backend_type("S3://BUCKET/KEY")
            == StorageBackendType.S3
        )

    def test_detect_backend_type_gcs(self):
        """Test backend type detection for GCS paths."""
        assert (
            BaseStorageBackend.detect_backend_type("gs://bucket/key")
            == StorageBackendType.GCS
        )

    def test_detect_backend_type_http(self):
        """Test backend type detection for HTTP paths."""
        assert (
            BaseStorageBackend.detect_backend_type("https://example.com/file")
            == StorageBackendType.HTTP
        )
        assert (
            BaseStorageBackend.detect_backend_type("http://example.com/file")
            == StorageBackendType.HTTP
        )

    def test_detect_backend_type_azure(self):
        """Test backend type detection for Azure Blob paths."""
        assert (
            BaseStorageBackend.detect_backend_type(
                "https://account.blob.core.windows.net/container/blob"
            )
            == StorageBackendType.AZURE_BLOB
        )

    def test_detect_backend_type_ftp(self):
        """Test backend type detection for FTP paths."""
        assert (
            BaseStorageBackend.detect_backend_type("ftp://server/file")
            == StorageBackendType.FTP
        )

    def test_detect_backend_type_filesystem(self):
        """Test backend type detection for filesystem paths."""
        assert (
            BaseStorageBackend.detect_backend_type("/path/to/file")
            == StorageBackendType.FILESYSTEM
        )
        assert (
            BaseStorageBackend.detect_backend_type("./relative/path")
            == StorageBackendType.FILESYSTEM
        )
        assert (
            BaseStorageBackend.detect_backend_type("file.txt")
            == StorageBackendType.FILESYSTEM
        )

    def test_compute_hash(self):
        """Test content hash computation."""
        content = b"test content"
        hash1 = BaseStorageBackend.compute_hash(content)
        hash2 = BaseStorageBackend.compute_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_path_hash(self):
        """Test path hash computation."""
        path = "/path/to/file.txt"
        hash1 = BaseStorageBackend.compute_path_hash(path)
        hash2 = BaseStorageBackend.compute_path_hash(path)
        assert hash1 == hash2
        assert len(hash1) == 32  # Truncated hash


# =============================================================================
# FilesystemBackend Tests
# =============================================================================


class TestFilesystemBackend:
    """Tests for FilesystemBackend."""

    @pytest.mark.asyncio
    async def test_initialize_with_base_path(self, filesystem_backend, temp_directory):
        """Test initialization with base path."""
        await filesystem_backend.initialize({"base_path": temp_directory})
        assert filesystem_backend.base_path == Path(temp_directory)

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, filesystem_backend):
        """Test initialization creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "new_dir"
            await filesystem_backend.initialize({"base_path": str(new_path)})
            assert new_path.exists()

    @pytest.mark.asyncio
    async def test_initialize_empty_config(self, filesystem_backend):
        """Test initialization with empty config."""
        await filesystem_backend.initialize({})
        assert filesystem_backend.base_path is None

    @pytest.mark.asyncio
    async def test_exists_true(self, filesystem_backend, temp_directory):
        """Test exists returns True for existing file."""
        test_file = Path(temp_directory) / "test.txt"
        result = await filesystem_backend.exists(str(test_file))
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, filesystem_backend, temp_directory):
        """Test exists returns False for non-existing file."""
        result = await filesystem_backend.exists(
            str(Path(temp_directory) / "nonexistent.txt")
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metadata_existing_file(self, filesystem_backend, temp_directory):
        """Test get_metadata for existing file."""
        test_file = Path(temp_directory) / "test.txt"
        metadata = await filesystem_backend.get_metadata(str(test_file))

        assert metadata.exists is True
        assert metadata.origin_type == OriginType.FILE
        assert metadata.size > 0
        assert metadata.last_modified is not None
        assert metadata.content_type == "text/plain"
        assert metadata.metadata["filename"] == "test.txt"
        assert metadata.metadata["extension"] == ".txt"
        assert metadata.metadata["is_file"] is True

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent_file(
        self, filesystem_backend, temp_directory
    ):
        """Test get_metadata for non-existing file."""
        metadata = await filesystem_backend.get_metadata(
            str(Path(temp_directory) / "nonexistent.txt")
        )

        assert metadata.exists is False
        assert metadata.origin_type == OriginType.FILE

    @pytest.mark.asyncio
    async def test_read_file(self, filesystem_backend, temp_directory):
        """Test reading file content."""
        test_file = Path(temp_directory) / "test.txt"
        content = await filesystem_backend.read(str(test_file))

        assert content.content == b"Hello, World!"
        assert content.size == 13
        assert content.content_type == "text/plain"
        assert content.metadata["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_read_with_config(self, filesystem_backend, temp_directory):
        """Test reading file with custom config."""
        test_file = Path(temp_directory) / "test.txt"
        config = StorageConfig(chunk_size=4096)
        content = await filesystem_backend.read(str(test_file), config)

        assert content.content == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_read_stream(self, filesystem_backend, temp_directory):
        """Test streaming file content."""
        test_file = Path(temp_directory) / "test.txt"
        chunks = []

        async for chunk in filesystem_backend.read_stream(str(test_file)):
            chunks.append(chunk)

        assert b"".join(chunks) == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_read_stream_with_config(self, filesystem_backend, temp_directory):
        """Test streaming with custom chunk size."""
        test_file = Path(temp_directory) / "test.txt"
        config = StorageConfig(chunk_size=5)
        chunks = []

        async for chunk in filesystem_backend.read_stream(str(test_file), config):
            chunks.append(chunk)

        assert b"".join(chunks) == b"Hello, World!"
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_list_contents_non_recursive(
        self, filesystem_backend, temp_directory
    ):
        """Test listing directory contents non-recursively."""
        results = await filesystem_backend.list_contents(
            temp_directory, recursive=False
        )

        filenames = [Path(r.origin_path).name for r in results]
        assert "test.txt" in filenames
        # nested.md should not be included in non-recursive listing
        assert "nested.md" not in filenames

    @pytest.mark.asyncio
    async def test_list_contents_recursive(self, filesystem_backend, temp_directory):
        """Test listing directory contents recursively."""
        results = await filesystem_backend.list_contents(temp_directory, recursive=True)

        filenames = [Path(r.origin_path).name for r in results]
        assert "test.txt" in filenames
        assert "nested.md" in filenames

    @pytest.mark.asyncio
    async def test_list_contents_with_pattern(self, filesystem_backend, temp_directory):
        """Test listing with glob pattern."""
        results = await filesystem_backend.list_contents(
            temp_directory, recursive=True, pattern="*.txt"
        )

        filenames = [Path(r.origin_path).name for r in results]
        assert "test.txt" in filenames
        assert "nested.md" not in filenames

    @pytest.mark.asyncio
    async def test_list_contents_not_directory(
        self, filesystem_backend, temp_directory
    ):
        """Test listing contents of a file returns empty list."""
        test_file = Path(temp_directory) / "test.txt"
        results = await filesystem_backend.list_contents(str(test_file))
        assert results == []

    @pytest.mark.asyncio
    async def test_resolve_path_with_base(self, filesystem_backend, temp_directory):
        """Test path resolution with base path."""
        await filesystem_backend.initialize({"base_path": temp_directory})
        resolved = filesystem_backend._resolve_path("test.txt")
        assert resolved == Path(temp_directory) / "test.txt"

    # @pytest.mark.asyncio
    # async def test_resolve_path_absolute(self, filesystem_backend, temp_directory):
    #     """Test absolute path resolution."""
    #     await filesystem_backend.initialize({"base_path": "/some/base"})
    #     absolute_path = Path(temp_directory) / "test.txt"
    #     resolved = filesystem_backend._resolve_path(str(absolute_path))
    #     assert resolved == absolute_path
    #


# =============================================================================
# S3Backend Tests
# =============================================================================


class TestS3Backend:
    """Tests for S3Backend."""

    def test_parse_s3_path(self, s3_backend):
        """Test S3 path parsing."""
        bucket, key = s3_backend._parse_s3_path("s3://my-bucket/path/to/file.txt")
        assert bucket == "my-bucket"
        assert key == "path/to/file.txt"

    def test_parse_s3_path_no_key(self, s3_backend):
        """Test S3 path parsing with no key."""
        bucket, key = s3_backend._parse_s3_path("s3://my-bucket")
        assert bucket == "my-bucket"
        assert key == ""

    # @pytest.mark.asyncio
    # async def test_initialize_success(self, s3_backend):
    #     """Test successful S3 initialization."""
    #     with patch(
    #         "FOSRABack.src.retrieval.services.origin_service.aiobotocore"
    #     ) as mock_aiobotocore:
    #         mock_session = MagicMock()
    #         mock_client = AsyncMock()
    #         mock_session.get_session.return_value = mock_session
    #         mock_session.create_client.return_value.__aenter__ = AsyncMock(
    #             return_value=mock_client
    #         )
    #         mock_aiobotocore.session = mock_session
    #
    #         await s3_backend.initialize(
    #             {
    #                 "aws_access_key_id": "test_key",
    #                 "aws_secret_access_key": "test_secret",
    #                 "region_name": "us-west-2",
    #             }
    #         )
    #
    #         assert s3_backend._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_import_error(self, s3_backend):
        """Test S3 initialization with missing aiobotocore."""
        with patch.dict(
            "sys.modules", {"aiobotocore": None, "aiobotocore.session": None}
        ):
            with pytest.raises(Exception):
                await s3_backend.initialize({})

    @pytest.mark.asyncio
    async def test_exists_not_initialized(self, s3_backend):
        """Test exists returns False when not initialized."""
        result = await s3_backend.exists("s3://bucket/key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, s3_backend):
        """Test exists returns True for existing object."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()
        s3_backend.client.head_object = AsyncMock(return_value={})

        result = await s3_backend.exists("s3://bucket/key")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, s3_backend):
        """Test exists returns False for non-existing object."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()
        s3_backend.client.head_object = AsyncMock(side_effect=Exception("Not found"))

        result = await s3_backend.exists("s3://bucket/key")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metadata_not_initialized(self, s3_backend):
        """Test get_metadata when not initialized."""
        metadata = await s3_backend.get_metadata("s3://bucket/key")
        assert metadata.exists is False

    @pytest.mark.asyncio
    async def test_get_metadata_success(self, s3_backend):
        """Test get_metadata for existing object."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()
        s3_backend.client.head_object = AsyncMock(
            return_value={
                "ContentLength": 1024,
                "LastModified": datetime.now(),
                "ContentType": "text/plain",
                "Metadata": {"custom": "value"},
            }
        )

        metadata = await s3_backend.get_metadata("s3://bucket/key")
        assert metadata.exists is True
        assert metadata.size == 1024
        assert metadata.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self, s3_backend):
        """Test get_metadata for non-existing object."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()
        s3_backend.client.head_object = AsyncMock(side_effect=Exception("Not found"))

        metadata = await s3_backend.get_metadata("s3://bucket/key")
        assert metadata.exists is False

    @pytest.mark.asyncio
    async def test_read_not_initialized(self, s3_backend):
        """Test read raises when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await s3_backend.read("s3://bucket/key")

    @pytest.mark.asyncio
    async def test_read_success(self, s3_backend):
        """Test successful read from S3."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()

        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=b"test content")
        mock_body.__aenter__ = AsyncMock(return_value=mock_body)
        mock_body.__aexit__ = AsyncMock()

        s3_backend.client.get_object = AsyncMock(
            return_value={
                "Body": mock_body,
                "ContentType": "text/plain",
                "LastModified": datetime.now(),
            }
        )

        content = await s3_backend.read("s3://bucket/key")
        assert content.content == b"test content"

    @pytest.mark.asyncio
    async def test_read_stream_not_initialized(self, s3_backend):
        """Test read_stream raises when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in s3_backend.read_stream("s3://bucket/key"):
                pass

    @pytest.mark.asyncio
    async def test_read_stream_success(self, s3_backend):
        """Test successful streaming from S3."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()

        call_count = {"value": 0}

        async def mock_read(size):
            if call_count["value"] == 0:
                call_count["value"] += 1
                return b"chunk"
            return b""

        mock_body = AsyncMock()
        mock_body.read = mock_read
        mock_body.__aenter__ = AsyncMock(return_value=mock_body)
        mock_body.__aexit__ = AsyncMock()

        s3_backend.client.get_object = AsyncMock(return_value={"Body": mock_body})

        chunks = []
        async for chunk in s3_backend.read_stream("s3://bucket/key"):
            chunks.append(chunk)

        assert b"chunk" in chunks

    @pytest.mark.asyncio
    async def test_list_contents_not_initialized(self, s3_backend):
        """Test list_contents returns empty when not initialized."""
        results = await s3_backend.list_contents("s3://bucket/prefix")
        assert results == []

    @pytest.mark.asyncio
    async def test_list_contents_success(self, s3_backend):
        """Test successful listing from S3."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()

        async def mock_paginate(**kwargs):
            yield {
                "Contents": [
                    {"Key": "file1.txt", "Size": 100, "LastModified": datetime.now()},
                    {"Key": "file2.txt", "Size": 200, "LastModified": datetime.now()},
                ]
            }

        mock_paginator = MagicMock()
        mock_paginator.paginate = mock_paginate
        s3_backend.client.get_paginator = MagicMock(return_value=mock_paginator)

        results = await s3_backend.list_contents("s3://bucket/prefix", recursive=True)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_contents_with_pattern(self, s3_backend):
        """Test listing with pattern filter."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()

        async def mock_paginate(**kwargs):
            yield {
                "Contents": [
                    {"Key": "file1.txt", "Size": 100, "LastModified": datetime.now()},
                    {"Key": "file2.md", "Size": 200, "LastModified": datetime.now()},
                ]
            }

        mock_paginator = MagicMock()
        mock_paginator.paginate = mock_paginate
        s3_backend.client.get_paginator = MagicMock(return_value=mock_paginator)

        results = await s3_backend.list_contents("s3://bucket/prefix", pattern="*.txt")
        assert len(results) == 1
        assert "file1.txt" in results[0].origin_path

    @pytest.mark.asyncio
    async def test_close(self, s3_backend):
        """Test closing S3 client."""
        s3_backend._initialized = True
        s3_backend.client = AsyncMock()
        s3_backend.client.__aexit__ = AsyncMock()

        await s3_backend.close()
        assert s3_backend._initialized is False


# =============================================================================
# HTTPBackend Tests
# =============================================================================


class TestHTTPBackend:
    """Tests for HTTPBackend."""

    # @pytest.mark.asyncio
    # async def test_initialize_success(self, http_backend):
    #     """Test successful HTTP initialization."""
    #     with patch("FOSRABack.src.retrieval.services.origin_service.httpx") as mock_httpx:
    #         mock_client = AsyncMock()
    #         mock_httpx.AsyncClient.return_value = mock_client
    #
    #         await http_backend.initialize({"timeout": 60, "headers": {"User-Agent": "test"}})
    #         assert http_backend._initialized is True

    @pytest.mark.asyncio
    async def test_exists_not_initialized(self, http_backend):
        """Test exists returns False when not initialized."""
        result = await http_backend.exists("https://example.com/file")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, http_backend):
        """Test exists returns True for accessible URL."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        http_backend.client.head = AsyncMock(return_value=mock_response)

        result = await http_backend.exists("https://example.com/file")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, http_backend):
        """Test exists returns False for inaccessible URL."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        http_backend.client.head = AsyncMock(side_effect=Exception("Connection error"))

        result = await http_backend.exists("https://example.com/file")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metadata_not_initialized(self, http_backend):
        """Test get_metadata when not initialized."""
        metadata = await http_backend.get_metadata("https://example.com/file")
        assert metadata.exists is False
        assert metadata.origin_type == OriginType.CRAWLED_URL

    @pytest.mark.asyncio
    async def test_get_metadata_success(self, http_backend):
        """Test successful metadata retrieval."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-length": "1024",
            "content-type": "text/html",
        }
        http_backend.client.head = AsyncMock(return_value=mock_response)

        metadata = await http_backend.get_metadata("https://example.com/file")
        assert metadata.exists is True
        assert metadata.size == 1024
        assert metadata.content_type == "text/html"

    @pytest.mark.asyncio
    async def test_get_metadata_error(self, http_backend):
        """Test get_metadata handles errors."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        http_backend.client.head = AsyncMock(side_effect=Exception("Error"))

        metadata = await http_backend.get_metadata("https://example.com/file")
        assert metadata.exists is False

    @pytest.mark.asyncio
    async def test_read_not_initialized(self, http_backend):
        """Test read raises when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await http_backend.read("https://example.com/file")

    @pytest.mark.asyncio
    async def test_read_success(self, http_backend):
        """Test successful HTTP read."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()
        http_backend.client.get = AsyncMock(return_value=mock_response)

        content = await http_backend.read("https://example.com/file")
        assert content.content == b"test content"
        assert content.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_read_stream_not_initialized(self, http_backend):
        """Test read_stream raises when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in http_backend.read_stream("https://example.com/file"):
                pass

    @pytest.mark.asyncio
    async def test_read_stream_success(self, http_backend):
        """Test successful HTTP streaming."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()

        async def mock_aiter_bytes(chunk_size):
            yield b"chunk1"
            yield b"chunk2"

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        http_backend.client.stream = MagicMock(return_value=mock_response)

        chunks = []
        async for chunk in http_backend.read_stream("https://example.com/file"):
            chunks.append(chunk)

        assert chunks == [b"chunk1", b"chunk2"]

    @pytest.mark.asyncio
    async def test_list_contents_not_supported(self, http_backend):
        """Test list_contents returns empty for HTTP."""
        http_backend._initialized = True
        results = await http_backend.list_contents("https://example.com/")
        assert results == []

    @pytest.mark.asyncio
    async def test_close(self, http_backend):
        """Test closing HTTP client."""
        http_backend._initialized = True
        http_backend.client = AsyncMock()
        http_backend.client.aclose = AsyncMock()

        await http_backend.close()
        assert http_backend._initialized is False


# =============================================================================
# OriginService Tests
# =============================================================================


class TestOriginService:
    """Tests for OriginService."""

    def test_get_backend(self, origin_service):
        """Test getting a backend by type."""
        backend = origin_service.get_backend(StorageBackendType.FILESYSTEM)
        assert backend is not None
        assert isinstance(backend, FilesystemBackend)

    def test_get_backend_default(self, origin_service):
        """Test getting default backend."""
        backend = origin_service.get_backend()
        assert backend is not None
        assert isinstance(backend, FilesystemBackend)

    def test_get_backend_cached(self, origin_service):
        """Test backend caching."""
        backend1 = origin_service.get_backend(StorageBackendType.FILESYSTEM)
        backend2 = origin_service.get_backend(StorageBackendType.FILESYSTEM)
        assert backend1 is backend2

    def test_detect_backend_filesystem(self, origin_service):
        """Test backend detection for filesystem paths."""
        backend = origin_service.detect_backend("/path/to/file")
        assert isinstance(backend, FilesystemBackend)

    def test_detect_backend_http(self, origin_service):
        """Test backend detection for HTTP paths."""
        backend = origin_service.detect_backend("https://example.com/file")
        assert isinstance(backend, HTTPBackend)

    def test_detect_backend_s3(self, origin_service):
        """Test backend detection for S3 paths."""
        backend = origin_service.detect_backend("s3://bucket/key")
        assert isinstance(backend, S3Backend)

    @pytest.mark.asyncio
    async def test_initialize_backend(self, origin_service):
        """Test initializing a backend."""
        backend = await origin_service.initialize_backend(
            StorageBackendType.FILESYSTEM, {"base_path": "/tmp"}
        )
        assert backend is not None

    def test_get_available_backends(self, origin_service):
        """Test getting available backends."""
        backends = origin_service.get_available_backends()
        assert StorageBackendType.FILESYSTEM in backends

    @pytest.mark.asyncio
    async def test_get_or_create_origin_new(self, origin_service, mock_session):
        """Test creating new origin."""
        # Mock no existing origin
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute.return_value = mock_result

        origin = await origin_service.get_or_create_origin(
            path="/path/to/file.txt",
            origin_type=OriginType.FILE,
            name="test_file",
            connector_type=ConnectorType.FILE_UPLOAD,
        )

        assert origin is not None
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_origin_existing(self, origin_service, mock_session):
        """Test returning existing origin."""
        existing_origin = OriginORM(
            source_hash="test_hash",
            name="existing",
            origin_path="/path/to/file.txt",
            origin_type=OriginType.FILE,
        )

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = existing_origin
        mock_session.execute.return_value = mock_result

        origin = await origin_service.get_or_create_origin(
            path="/path/to/file.txt",
            origin_type=OriginType.FILE,
        )

        assert origin is existing_origin
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_origin_string_path(self, origin_service, temp_directory):
        """Test reading origin with string path."""
        test_file = str(Path(temp_directory) / "test.txt")
        content = await origin_service.read_origin(test_file)
        assert content.content == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_read_origin_orm(self, origin_service, temp_directory):
        """Test reading origin with ORM object."""
        test_file = str(Path(temp_directory) / "test.txt")
        origin = OriginORM(
            source_hash="test",
            name="test",
            origin_path=test_file,
            origin_type=OriginType.FILE,
        )

        content = await origin_service.read_origin(origin)
        assert content.content == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_read_origin_no_backend(self, origin_service):
        """Test read_origin raises for unsupported path."""
        # Manually clear the registry for a type that's not implemented
        origin_service._backends.clear()

        with patch.object(BaseStorageBackend, "_registry", {}):
            with pytest.raises(ValueError, match="No backend available"):
                await origin_service.read_origin("gs://bucket/key")

    @pytest.mark.asyncio
    async def test_read_origin_stream(self, origin_service, temp_directory):
        """Test streaming origin content."""
        test_file = str(Path(temp_directory) / "test.txt")
        chunks = []

        async for chunk in origin_service.read_origin_stream(test_file):
            chunks.append(chunk)

        assert b"".join(chunks) == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_read_origin_stream_no_backend(self, origin_service):
        """Test read_origin_stream raises for unsupported path."""
        origin_service._backends.clear()

        with patch.object(BaseStorageBackend, "_registry", {}):
            with pytest.raises(ValueError, match="No backend available"):
                async for _ in origin_service.read_origin_stream("gs://bucket/key"):
                    pass

    @pytest.mark.asyncio
    async def test_get_origin_metadata(self, origin_service, temp_directory):
        """Test getting origin metadata."""
        test_file = str(Path(temp_directory) / "test.txt")
        metadata = await origin_service.get_origin_metadata(test_file)

        assert metadata.exists is True
        assert metadata.size > 0

    @pytest.mark.asyncio
    async def test_get_origin_metadata_no_backend(self, origin_service):
        """Test get_origin_metadata with no backend."""
        origin_service._backends.clear()

        with patch.object(BaseStorageBackend, "_registry", {}):
            metadata = await origin_service.get_origin_metadata("gs://bucket/key")
            assert metadata.exists is False

    @pytest.mark.asyncio
    async def test_list_origins(self, origin_service, temp_directory):
        """Test listing origins in directory."""
        results = await origin_service.list_origins(temp_directory, recursive=True)

        filenames = [Path(r.origin_path).name for r in results]
        assert "test.txt" in filenames

    @pytest.mark.asyncio
    async def test_list_origins_no_backend(self, origin_service):
        """Test list_origins with no backend."""
        origin_service._backends.clear()

        with patch.object(BaseStorageBackend, "_registry", {}):
            results = await origin_service.list_origins("gs://bucket/prefix")
            assert results == []

    @pytest.mark.asyncio
    async def test_close_all(self, origin_service):
        """Test closing all backends."""
        # Get some backends
        origin_service.get_backend(StorageBackendType.FILESYSTEM)
        origin_service.get_backend(StorageBackendType.HTTP)

        assert len(origin_service._backends) > 0

        await origin_service.close_all()

        assert len(origin_service._backends) == 0


# =============================================================================
# StorageConfig and Data Classes Tests
# =============================================================================


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StorageConfig()
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.chunk_size == 8192
        assert config.backend_options == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StorageConfig(
            timeout_seconds=60,
            max_retries=5,
            chunk_size=4096,
            backend_options={"key": "value"},
        )
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
        assert config.chunk_size == 4096
        assert config.backend_options == {"key": "value"}


class TestOriginContent:
    """Tests for OriginContent dataclass."""

    def test_creation(self):
        """Test creating OriginContent."""
        content = OriginContent(
            source_hash="abc123",
            content=b"test",
            content_type="text/plain",
            size=4,
        )
        assert content.source_hash == "abc123"
        assert content.content == b"test"
        assert content.content_type == "text/plain"
        assert content.size == 4


class TestOriginMetadata:
    """Tests for OriginMetadata dataclass."""

    def test_creation(self):
        """Test creating OriginMetadata."""
        metadata = OriginMetadata(
            source_hash="abc123",
            origin_path="/path/to/file",
            origin_type=OriginType.FILE,
            exists=True,
            size=1024,
        )
        assert metadata.source_hash == "abc123"
        assert metadata.origin_path == "/path/to/file"
        assert metadata.origin_type == OriginType.FILE
        assert metadata.exists is True
        assert metadata.size == 1024
