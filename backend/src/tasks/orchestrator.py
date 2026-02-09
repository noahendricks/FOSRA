from __future__ import annotations

from datetime import datetime
import time
from typing import TYPE_CHECKING

import logfire
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.src.api.lifecycle import Infrastructure
from backend.src.domain.exceptions import (
    InfrastructureError,
    RAGError,
    SourceRetrievalError,
    VectorSearchError,
)
from backend.src.services.retrieval.impls._utils import (
    deduplicate_results,
    group_by_source,
    rerank_results,
    transform_vector_results,
)
from backend.src.storage.models import SourceORM
from backend.src.domain.schemas import (
    ChunkerConfig,
    EmbedderConfig,
    ParserConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RetrievedResult,
)

from backend.src.api.schemas import SourceResponseDeep
from backend.src.storage.utils.converters import pydantic_to_domain
from backend.src.tasks.processing import chunk_sources
from backend.src.tasks.processing import embed_documents
from backend.src.tasks.processing import parse_files
from backend.src.tasks.storing import store_file_vectors

from backend.src.api.request_context import RequestContext

if TYPE_CHECKING:
    pass


from .broker import broker
