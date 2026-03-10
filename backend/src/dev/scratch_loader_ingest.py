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
from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    UserPreferences,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import (Doc, MDNFile, PDFMetadata, TextMetadata)
from backend.src.services.conversation import llm_service
from backend.src.services.conversation.conversation_service import ConversationService
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.processing.loader_service import (LoaderService, to_bytes)
from backend.src.services.retrieval.vector_service import VectorService

# docs = joblib.load(".cache/docs.pkl")

# working on chunker right now
md_bytes = to_bytes(
    "/home/roccoluxe/Documents/docs/09-frontend-ui/tsquery/reference/querying/QueryClient.md"
)

md_blob = Blob.from_data(data=md_bytes)

pdf_bytes = to_bytes("/home/roccoluxe/Documents/Misc/MakingMusic_DennisDeSantis.pdf")

pdf_blob = Blob.from_data(data=pdf_bytes)

mock_mdn_pdf = MDNFile(
    media_type="application/pdf",
    type=pdf_blob.mimetype or "",
    name=str(pdf_blob.path),
    size=0,
    bytes=pdf_bytes,
    webkit_relative_path=pdf_blob.source,
)

mock_mdn_md = MDNFile(
    media_type="text/markdown",
    type=md_blob.mimetype or "",
    name=str(md_blob.path),
    size=0,
    bytes=md_blob.data,
    webkit_relative_path=md_blob.source,
)

result: list[Doc] = LoaderService.parse_files([mock_mdn_md])
