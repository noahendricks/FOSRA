from __future__ import annotations

import asyncio
import mimetypes
import pprint
from pathlib import Path
from pprint import pp
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from loguru import logger
from ulid import ULID

from backend.src.domain.enums import VectorStoreType
from backend.src.settings import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import (Doc, MDNFile, PDFMetadata, TextMetadata)
from backend.src.services.conversation import llm_service
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.processing.hi_chunk import HiChunk, HiChunkStructurer
from backend.src.services.processing.loader_service import (LoaderService, to_bytes)
from backend.src.services.processing.utils.loader import code_mimes
from backend.src.services.retrieval.vector_service import VectorService

# docs = joblib.load(".cache/docs.pkl")

# working on chunker right now

py_path = "/home/roccoluxe/FOSRA/backend/src/services/processing/embedder_service.py"

md_path = "/home/roccoluxe/Documents/docs/04-languages-types/typescript/02-type-system/Basic Types.md"

from content_types import get_content_type as get_mime

pdf_path = "/home/roccoluxe/Documents/Misc/MakingMusic_DennisDeSantis.pdf"
pdf_bytes = to_bytes("/home/roccoluxe/Documents/Misc/MakingMusic_DennisDeSantis.pdf")

mime = get_mime(pdf_path)

# i = LoaderService().parse_files([py_path])

result: list[Doc] = LoaderService._parse_files([md_path])

print(result)

structurer = HiChunkStructurer(config=ChunkerConfig())


chunks = asyncio.run(
    ChunkerService()._chunk_text(doc=result[0], config=ChunkerConfig())
)


embedded_chunks = asyncio.run(
    EmbedderService().embed_chunks(chunks=chunks, config=EmbedderConfig())
)

point_ids = asyncio.run(
    VectorService.upsert(
        chunks=embedded_chunks,
        config=VectorStoreConfig(),
        embed_config=EmbedderConfig(),
    )
)

print(point_ids)
