from __future__ import annotations
from typing import BinaryIO
import markitdown


from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.api.schemas.api_schemas import FilePart
from backend.src.domain.schemas import FileContent, FileMetadata, StorageConfig


from loguru import logger

from .broker import broker

from blake3 import blake3



@broker.task
async def ingest_files(
    files: list[FilePart], session: AsyncSession, storage_config: StorageConfig
) -> list[FileContent]:

    files_out = []
    try:
        for f in files:
            context = f"{f.filename}, {f.type}"
            hash = blake3(f.bytes, derive_key_context=context).hexdigest()
            logger.debug(hash)

            from markitdown import MarkItDown
            import base64

            # WARN: AFTER BINARY DECODE, FILE TYPE CHECK NECESSARY

            decode = base64.b64decode(f.bytes)
            md = MarkItDown()

            files_out.append(
                FileContent(
                    file_path="path",
                    file_name=f.filename,
                    content=decode.decode("utf-8", "strict"),
                    file_hash=hash,
                    metadata=FileMetadata(),
                )
            )
            print()

        return files_out

    except Exception as e:
        raise e
