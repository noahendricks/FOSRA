from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

import logfire
from loguru import logger

from backend.src.api.schemas.api_schemas import FilePart
from backend.src.domain.exceptions import (
    FileNotFoundError,
    FileReadError,
    FileListError,
    UnsupportedBackendError,
    StorageBackendNotFoundError,
)
from backend.src.domain.schemas import (
    FileContent,
    FileMetadata,
    StorageConfig,
)


from backend.src.domain.enums import StorageBackendType
from backend.src.services.retrieval.impls.backends import BaseStorageBackend

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class FileService:
    def get_backend(
        backend_type: StorageBackendType,
        session: AsyncSession,
    ) -> BaseStorageBackend:
        backend = BaseStorageBackend.get_backend(backend_type, session)

        if not backend:
            available = BaseStorageBackend.get_available_backends()
            logger.error(f"Storage backend not found: {backend_type}")
            raise StorageBackendNotFoundError(
                backend_type=backend_type.value,
                remediation=f"Available backends: {', '.join(available)}",
            )

        logger.debug(f"Retrieved storage backend: {backend_type.value}")
        return backend

    @staticmethod
    def get_available_backends() -> list[str]:
        return BaseStorageBackend.get_available_backends()

    @staticmethod
    def detect_backend(
        file_path: str,
        session: AsyncSession,
    ) -> BaseStorageBackend:
        backend_type = BaseStorageBackend.detect_backend_type(file_path)
        logger.debug(f"Detected backend type {backend_type.value} for {file_path}")

        try:
            return FileService.get_backend(backend_type, session)
        except StorageBackendNotFoundError:
            available = FileService.get_available_backends()
            raise UnsupportedBackendError(
                file_path=file_path,
                available_backends=available,
            )

    @staticmethod
    async def initialize_backend(
        backend_type: StorageBackendType,
        config: dict[str, Any],
        session: AsyncSession,
    ) -> BaseStorageBackend:
        """Initialize a storage backend with configuration.

        Returns:
            Initialized storage backend
        """
        logger.info(f"Initializing {backend_type.value} storage backend")

        backend = FileService.get_backend(backend_type, session)
        await backend.initialize(config)

        logger.success(f"Successfully initialized {backend_type.value} backend")
        return backend

    @staticmethod
    async def file_exists(
        file_path: str,
        session: AsyncSession,
        backend_type: StorageBackendType | None = None,
    ) -> bool:
        logger.debug(f"Checking existence of file: {file_path}")

        try:
            if backend_type:
                backend = FileService.get_backend(backend_type, session)
            else:
                backend = FileService.detect_backend(file_path, session)

            exists = await backend.exists(file_path)
            logger.debug(f"File {file_path} exists: {exists}")
            return exists

        except Exception as e:
            logger.warning(f"Error checking file existence for {file_path}: {e}")
            return False

    @staticmethod
    async def get_file_metadata(
        file_path: str,
        session: AsyncSession,
        backend_type: StorageBackendType | None = None,
    ) -> FileMetadata:
        logger.debug(f"Retrieving metadata for file: {file_path}")
        try:
            if backend_type:
                backend = FileService.get_backend(backend_type, session)
            else:
                backend = FileService.detect_backend(file_path, session)

            metadata = await backend.get_metadata(file_path)

            logger.info(f"Retrieved metadata for {file_path}: size={metadata.size}")

            return metadata

        except (StorageBackendNotFoundError, UnsupportedBackendError):
            raise
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_path}: {e}")
            raise FileReadError(
                file_path=file_path,
                reason=str(e),
                backend_type=backend_type.value if backend_type else None,
            ) from e

    @staticmethod
    async def read_file(
        file: FilePart,
        session: AsyncSession,
        config: StorageConfig | None = None,
        backend_type: StorageBackendType | None = None,
    ) -> FileContent:
        logger.info(f"Reading file: {file_path}")
        logger.info(f"Entering read_file")

        try:
            if backend_type:
                backend = FileService.get_backend(backend_type, session)
            else:
                backend = FileService.detect_backend(file_path, session)

            logger.debug(f"Verifying existence of file before read: {file_path}")

            exists = await backend.exists(file_path)

            if not exists:
                raise FileNotFoundError(
                    file_path=file_path,
                    backend_type=backend.backend_type.value,
                )

            metadata = await FileService.get_file_metadata(
                file_path=file_path,
                session=session,
            )
            content: FileContent = await backend.read(file_path, config)

            logger.success(
                f"Successfully read file {file_path}: "
                f"{len(content.content)} bytes, "
                f"type={content.metadata.document_type if content else 'N/A'}"
            )

            return content

        except FileNotFoundError:
            raise
        except (StorageBackendNotFoundError, UnsupportedBackendError):
            raise
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise FileReadError(
                file_path=file_path,
                reason=str(e),
                backend_type=backend_type.value if backend_type else None,
            ) from e

    @staticmethod
    async def read_file_stream(
        file_path: str,
        session: AsyncSession,
        config: StorageConfig | None = None,
        backend_type: StorageBackendType | None = None,
    ) -> AsyncIterator[bytes]:
        logger.info(f"Starting stream for file: {file_path}")

        try:
            if backend_type:
                backend = FileService.get_backend(backend_type, session)
            else:
                backend = FileService.detect_backend(file_path, session)

            exists = await backend.exists(file_path)

            if not exists:
                raise FileNotFoundError(
                    file_path=file_path,
                    backend_type=backend.backend_type.value,
                )

            bytes_streamed = 0
            async for chunk in await backend.read_stream(file_path, config):
                bytes_streamed += len(chunk)
                yield chunk

            logger.success(
                f"Successfully streamed {bytes_streamed} bytes from {file_path}"
            )

        except FileNotFoundError:
            raise
        except (StorageBackendNotFoundError, UnsupportedBackendError):
            raise
        except Exception as e:
            logger.error(f"Failed to stream file {file_path}: {e}")
            raise FileReadError(
                file_path=file_path,
                reason=str(e),
                backend_type=backend_type.value if backend_type else None,
            ) from e

    @staticmethod
    async def list_files(
        path: str,
        session: AsyncSession,
        recursive: bool = False,
        pattern: str | None = None,
        backend_type: StorageBackendType | None = None,
    ) -> list[FileMetadata]:
        logger.info(
            f"Listing files at {path} (recursive={recursive}, pattern={pattern})"
        )

        try:
            if backend_type:
                backend = FileService.get_backend(backend_type, session)
            else:
                backend = FileService.detect_backend(path, session)

            files = await backend.list_contents(path, recursive, pattern)

            logger.success(f"Found {len(files)} files at {path}")
            return files

        except (StorageBackendNotFoundError, UnsupportedBackendError):
            raise
        except Exception as e:
            logger.error(f"Failed to list files at {path}: {e}")
            raise FileListError(
                path=path,
                reason=str(e),
                backend_type=backend_type.value if backend_type else None,
            ) from e

    @staticmethod
    @logfire.instrument("Processing batch of files")
    async def read_files_batch(
        file_paths: list[str],
        session: AsyncSession,
        config: StorageConfig | None = None,
        max_concurrent: int = 5,
    ) -> list[FileContent | None]:
        import asyncio

        if not file_paths:
            logger.warning("No file paths provided for batch read")
            return []

        logger.info(
            f"Reading {len(file_paths)} files in batch (max_concurrent={max_concurrent})"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_single(path: str) -> FileContent | None:
            """Read a single file with concurrency control."""
            async with semaphore:
                try:
                    return await FileService.read_file(
                        file_path=path,
                        session=session,
                        config=config,
                    )
                except Exception as e:
                    logger.error(f"Failed to read {path} in batch: {e}")
                    return None

        results = await asyncio.gather(
            *[read_single(path) for path in file_paths],
            return_exceptions=False,
        )

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch read completed: {successful}/{len(file_paths)} successful")

        return list(results)

    @staticmethod
    def compute_file_hash(content: bytes, file_path) -> str:
        return BaseStorageBackend.compute_hash(content, file_path)

    @staticmethod
    def compute_path_hash(file_path: str) -> str:
        return BaseStorageBackend.compute_path_hash(file_path)
