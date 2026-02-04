from __future__ import annotations

from datetime import datetime
import time
from typing import TYPE_CHECKING

import logfire
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from FOSRABack.src.api.lifecycle import Infrastructure
from FOSRABack.src.domain.exceptions import (
    InfrastructureError,
    RAGError,
    SourceRetrievalError,
    VectorSearchError,
)
from FOSRABack.src.services.retrieval.impls._utils import (
    deduplicate_results,
    group_by_source,
    rerank_results,
    transform_vector_results,
)
from FOSRABack.src.storage.models import SourceORM
from FOSRABack.src.domain.schemas import (
    ChunkerConfig,
    EmbedderConfig,
    ParserConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RetrievedResult,
)

from FOSRABack.src.api.schemas import SourceResponseDeep
from FOSRABack.src.storage.utils.converters import pydantic_to_domain
from FOSRABack.src.tasks.processing import chunk_sources
from FOSRABack.src.tasks.processing import embed_documents
from FOSRABack.src.tasks.processing import parse_files
from FOSRABack.src.tasks.storing import store_file_vectors

from FOSRABack.src.api.request_context import RequestContext

if TYPE_CHECKING:
    pass


from .broker import broker
