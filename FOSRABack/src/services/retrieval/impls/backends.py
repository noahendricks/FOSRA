from __future__ import annotations

import hashlib
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, ClassVar

import aiofiles
from FOSRABack.src.domain.enums import (
    DocumentType,
    FileType,
    OriginType,
    StorageBackendType,
)
from FOSRABack.src.domain.exceptions import (
    FileNotFoundError,
    FileReadError,
    StorageBackendInitializationError,
    StorageConnectionError,
    StorageCredentialsError,
)
from FOSRABack.src.domain.schemas import (
    FileContent,
    FileMetadata,
    StorageConfig,
)
from aiobotocore.session import AioBaseClient
from blake3 import blake3
from loguru import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# =============================================================================
# Base Storage Backend Interface
# =============================================================================


class BaseStorageBackend(ABC):
    _registry: ClassVar[dict[str, type[BaseStorageBackend]]] = {}

    def __init__(self, session: AsyncSession | None = None):
        self.session = session

    @classmethod
    def register(cls, backend_type: StorageBackendType):
        def decorator(backend_cls: type[BaseStorageBackend]):
            cls._registry[backend_type.value] = backend_cls
            return backend_cls

        return decorator

    @classmethod
    def get_backend(
        cls,
        backend_type: StorageBackendType,
        session: AsyncSession | None = None,
        **kwargs,
    ) -> BaseStorageBackend | None:
        backend_cls = cls._registry.get(backend_type.value)

        if backend_cls:
            return backend_cls(session=session, **kwargs)

        logger.warning(f"No backend found for type: {backend_type.value}")
        return None

    @classmethod
    def get_available_backends(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def detect_backend_type(cls, path: str) -> StorageBackendType:
        path_lower = path.lower()

        if path_lower.startswith("s3://"):
            return StorageBackendType.S3
        elif path_lower.startswith("gs://"):
            return StorageBackendType.GCS
        elif path_lower.startswith(("https://", "http://")):
            if "blob.core.windows.net" in path_lower:
                return StorageBackendType.AZURE_BLOB
            return StorageBackendType.HTTP
        elif path_lower.startswith("ftp://"):
            return StorageBackendType.FTP
        else:
            return StorageBackendType.FILESYSTEM

    # Backend identification (defined in subclasses)
    backend_type: ClassVar[StorageBackendType]
    display_name: ClassVar[str]

    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    async def get_metadata(self, path: str) -> FileMetadata:
        pass

    @abstractmethod
    async def read(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> FileContent:
        pass

    @abstractmethod
    async def read_stream(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> AsyncIterator[bytes]:
        pass

    @abstractmethod
    async def list_contents(
        self,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileMetadata]:
        pass

    async def close(self) -> None:
        pass

    @staticmethod
    def compute_hash(content: bytes, file_path: str) -> str:
        hasher = blake3()

        hasher.update(file_path.encode("utf-8"))

        hasher.update(content)

        return hasher.hexdigest()

    @staticmethod
    def compute_path_hash(path: str) -> str:
        hash = hashlib.sha256(path.encode()).hexdigest()[:32]

        return hash


# =============================================================================
# Filesystem Backend
# =============================================================================


@BaseStorageBackend.register(StorageBackendType.FILESYSTEM)
class FilesystemBackend(BaseStorageBackend):
    backend_type: ClassVar[StorageBackendType] = StorageBackendType.FILESYSTEM
    display_name: ClassVar[str] = "Local Filesystem"

    def __init__(self, session: AsyncSession | None = None, base_path: str = ""):
        super().__init__(session)
        self.base_path = Path(base_path) if base_path else None

    async def initialize(self, config: dict[str, Any]) -> None:
        if config.get("base_path"):
            self.base_path = Path(config["base_path"])

            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

                logger.info(f"Created base path: {self.base_path}")

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)

        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p

    async def exists(self, path: str) -> bool:
        resolved = self._resolve_path(path)
        exists = resolved.exists()
        logger.debug(f"File exists check for {path}: {exists}")
        return exists

    async def get_metadata(self, path: str) -> FileMetadata:
        resolved = self._resolve_path(path)

        if not resolved.exists():
            logger.debug(f"File not found while searching metadata for: {path}")
            pass

        try:
            stat = resolved.stat()
            print(resolved.name)

            content_type, _ = mimetypes.guess_type(str(resolved))
            print(str(resolved))

            document_type = DocumentType(content_type)

            metadata = FileMetadata(
                metadata_hash=self.compute_path_hash(path=path),
                file_type=FileType.FILE,
                file_name=resolved.name,
                size=stat.st_size,
                document_type=document_type,
                origin_type=OriginType.FILESYSTEM,
            )

            logger.debug(f"Retrieved metadata for {path}: size={metadata.size}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def read(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> FileContent:
        """Read file content."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(
                file_path=path,
                backend_type=self.backend_type.value,
            )

        try:
            async with aiofiles.open(resolved, "rb") as f:
                content = await f.read()

            document_type, _ = mimetypes.guess_type(str(resolved))

            if document_type and document_type in DocumentType:
                document_type = DocumentType(document_type)
            else:
                document_type = DocumentType.UNKNOWN

            metadata = await self.get_metadata(path)

            logger.info(f"Read {len(content)} bytes from {path}")
            file_hash = self.compute_hash(content=content, file_path=path)

            if not file_hash:
                logger.info(f"file hash generated for {path}: {file_hash}")

            return FileContent(
                file_path=path,
                file_name=resolved.name,
                content=content,
                file_hash=file_hash,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def read_stream(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream file content."""
        resolved = self._resolve_path(path)
        config = config or StorageConfig()

        if not resolved.exists():
            raise FileNotFoundError(
                file_path=path,
                backend_type=self.backend_type.value,
            )

        try:
            async with aiofiles.open(resolved, "rb") as f:
                while chunk := await f.read(config.chunk_size):
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to stream file {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def list_contents(
        self,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileMetadata]:
        """List directory contents."""
        resolved = self._resolve_path(path)

        if not resolved.is_dir():
            logger.warning(f"Path is not a directory: {path}")
            return []

        try:
            results = []

            if recursive:
                iterator = resolved.rglob(pattern or "*")
            else:
                iterator = resolved.glob(pattern or "*")

            for item in iterator:
                if item.is_file():
                    results.append(await self.get_metadata(str(item)))

            logger.info(f"Listed {len(results)} files in {path}")
            return results

        except Exception as e:
            logger.error(f"Failed to list contents of {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e


# =============================================================================
# S3 Backend
# =============================================================================


@BaseStorageBackend.register(StorageBackendType.S3)
class S3Backend(BaseStorageBackend):
    """AWS S3 storage backend."""

    backend_type: ClassVar[StorageBackendType] = StorageBackendType.S3
    display_name: ClassVar[str] = "Amazon S3"

    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self.client: AioBaseClient
        self._initialized = False

    async def initialize(self, config: dict[str, Any]) -> None:
        try:
            import aiobotocore.session

            if not config.get("aws_access_key_id") or not config.get(
                "aws_secret_access_key"
            ):
                raise StorageCredentialsError(
                    backend_type=self.backend_type.value,
                    reason="Missing AWS credentials",
                )

            aws_session = aiobotocore.session.get_session()
            self.client = await aws_session.create_client(
                "s3",
                aws_access_key_id=config.get("aws_access_key_id"),
                aws_secret_access_key=config.get("aws_secret_access_key"),
                region_name=config.get("region_name", "us-east-1"),
                endpoint_url=config.get("endpoint_url"),
            ).__aenter__()

            self._initialized = True
            logger.info("S3 client initialized successfully")

        except ImportError as e:
            raise StorageBackendInitializationError(
                backend_type=self.backend_type.value,
                reason=f"aiobotocore not installed: {e}",
                remediation="Install aiobotocore: pip install aiobotocore",
            ) from e
        except StorageCredentialsError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason=str(e),
            ) from e

    def _parse_s3_path(self, path: str) -> tuple[str, str]:
        """Parse s3://bucket/key path into bucket and key."""
        path = path.replace("s3://", "")

        parts = path.split("/", 1)

        bucket = parts[0]

        key = parts[1] if len(parts) > 1 else ""

        return bucket, key

    async def exists(self, path: str) -> bool:
        """Check if S3 object exists."""
        if not self._initialized:
            return False

        bucket, key = self._parse_s3_path(path)

        try:
            await self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    async def get_metadata(self, path: str) -> FileMetadata:
        """Get S3 object metadata."""
        bucket, key = self._parse_s3_path(path=path)

        if not self._initialized:
            return FileMetadata(
                metadata_hash=self.compute_path_hash(path=path),
                file_type=FileType.FILE,
            )

        try:
            response = await self.client.head_object(Bucket=bucket, Key=key)

            return FileMetadata(
                metadata_hash=self.compute_path_hash(path=path),
                file_type=FileType.FILE,
                size=response.get("ContentLength", 0),
            )

        except Exception:
            logger.debug(f"S3 object not found or error: {path}")
            return FileMetadata(
                metadata_hash=self.compute_path_hash(path=path),
                file_type=FileType.FILE,
            )

    async def read(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> FileContent:
        """Read S3 object content."""
        if not self._initialized:
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason="S3 client not initialized",
            )

        bucket, key = self._parse_s3_path(path)

        try:
            if not self.client:
                raise StorageConnectionError(
                    backend_type=self.backend_type.value,
                    reason="S3 client not initialized",
                )

            client = self.client
            response = await client.get_object(Bucket=bucket, Key=key)

            async with response["Body"] as stream:
                content = await stream.read()

            logger.info(f"Read {len(content)} bytes from S3: {path}")

            metadata = await self.get_metadata(path=path)

            return FileContent(
                file_name=key.split("/")[-1],
                file_path=path,
                content=content,
                file_hash=self.compute_path_hash(path),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to read S3 object {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def read_stream(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream S3 object content."""
        if not self._initialized:
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason="S3 client not initialized",
            )

        bucket, key = self._parse_s3_path(path)
        config = config or StorageConfig()

        try:
            response = await self.client.get_object(Bucket=bucket, Key=key)

            async with response["Body"] as stream:
                while chunk := await stream.read(config.chunk_size):
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to stream S3 object {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def list_contents(
        self,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileMetadata]:
        """List S3 bucket contents."""
        if not self._initialized:
            return []

        bucket, prefix = self._parse_s3_path(path)
        results = []

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            list_kwargs = {"Bucket": bucket, "Prefix": prefix}

            if not recursive:
                list_kwargs["Delimiter"] = "/"

            async for page in paginator.paginate(**list_kwargs):
                for obj in page.get("Contents", []):
                    key = obj["Key"]

                    # Apply pattern filter if provided
                    if pattern and not Path(key).match(pattern):
                        continue

                    # NOTE: Not fully implemented, doesn't get capture all metadata or get folders
                    results.append(
                        FileMetadata(
                            metadata_hash=self.compute_path_hash(path=key),
                            file_type=FileType.FILE,
                            size=obj["Size"],
                        )
                    )

            logger.info(f"Listed {len(results)} objects in S3: {path}")
            return results

        except Exception as e:
            logger.error(f"Failed to list S3 contents at {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def close(self) -> None:
        """Close S3 client."""
        if self.client:
            await self.client.__aexit__(None, None, None)
            self._initialized = False
            logger.debug("Closed S3 client")


# =============================================================================
# HTTP Backend
# =============================================================================


@BaseStorageBackend.register(StorageBackendType.HTTP)
class HTTPBackend(BaseStorageBackend):
    backend_type: ClassVar[StorageBackendType] = StorageBackendType.HTTP
    display_name: ClassVar[str] = "HTTP/HTTPS"

    def __init__(self, session: AsyncSession | None = None):
        super().__init__(session)
        self.client = None
        self._initialized = False

    async def initialize(self, config: dict[str, Any]) -> None:
        try:
            import httpx

            timeout = config.get("timeout", 30)
            headers = config.get("headers", {})

            self.client = httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
            )

            self._initialized = True
            logger.info("HTTP client initialized successfully")

        except ImportError as e:
            raise StorageBackendInitializationError(
                backend_type=self.backend_type.value,
                reason=f"httpx not installed: {e}",
                remediation="Install httpx: pip install httpx",
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize HTTP client: {e}")
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason=str(e),
            ) from e

    async def exists(self, path: str) -> bool:
        if not self._initialized:
            return False

        try:
            response = await self.client.head(path)
            return response.status_code == 200
        except Exception:
            return False

    async def get_metadata(self, path: str) -> FileMetadata:
        if not self._initialized:
            return FileMetadata(
                metadata_hash=self.compute_path_hash(path),
                file_type=FileType.CRAWLED_URL,
            )

        try:
            response = await self.client.head(path)
            content_length = response.headers.get("content-length")
            last_modified = response.headers.get("last-modified")
            content_type = response.headers.get("content-type")

            return FileMetadata(
                metadata_hash=self.compute_path_hash(path),
                file_type=FileType.CRAWLED_URL,
                size=int(content_length) if content_length else 0,
            )

        except Exception as e:
            logger.debug(f"HTTP metadata request failed for {path}: {e}")
            return FileMetadata(
                file_type=FileType.CRAWLED_URL,
            )

    async def read(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> FileContent:
        if not self._initialized:
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason="HTTP client not initialized",
            )

        try:
            response = await self.client.get(path)
            response.raise_for_status()

            logger.info(f"Read {len(response.content)} bytes from HTTP: {path}")

            return FileContent(
                file_path=path,
                file_name=path.split("/")[-1],
                content=response.content,
                file_hash=self.compute_path_hash(path),
                metadata=await self.get_metadata(path=path),
            )

        except Exception as e:
            logger.error(f"Failed to read HTTP resource {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def read_stream(
        self,
        path: str,
        config: StorageConfig | None = None,
    ) -> AsyncIterator[bytes]:
        if not self._initialized:
            raise StorageConnectionError(
                backend_type=self.backend_type.value,
                reason="HTTP client not initialized",
            )

        config = config or StorageConfig()

        try:
            async with self.client.stream("GET", path) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=config.chunk_size):
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to stream HTTP resource {path}: {e}")
            raise FileReadError(
                file_path=path,
                reason=str(e),
                backend_type=self.backend_type.value,
            ) from e

    async def list_contents(
        self,
        path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileMetadata]:
        logger.warning("list_contents not supported for HTTP backend")
        return []

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
            self._initialized = False
            logger.debug("Closed HTTP client")
