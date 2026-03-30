from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import AsyncTaskiqTask

from backend.src.domain.enums import FileSourceType, SourceType
from backend.src.settings import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import Chunk, Doc
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.vector_service import (
    CHUNKS_COLLECTION,
    VectorService,
)
from backend.src.storage.models import DocORM

from .broker import broker, get_infra


@broker.task(max_execution_time=300)
async def ingest_docs(
    docs: list[Doc],
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Ingest documents into Qdrant.

    Flow:
    1. Register docs in PostgreSQL (optional, if session_factory provided)
    2. Chunk via HiChunk (L1/L2)
    3. Embed all chunks
    4. Upsert to single collection

    Args:
        docs: List of Doc objects with page_content and metadata
        chunker_config: Chunker configuration
        embedder_config: Embedder configuration
        vector_config: Vector store configuration
        session_factory: Optional async session factory for PostgreSQL registration

    Returns:
        dict with counts: {chunks_upserted, docs_processed}
    """
    from backend.src.api.lifecycle import Infrastructure

    infra = get_infra()
    client = infra.qdrant_client

    if not isinstance(client, AsyncQdrantClient):
        raise RuntimeError("AsyncQdrantClient required for doc ingestion")

    # ensure collection exists
    await VectorService.ensure_collection(client, embedder_config)

    # step 1: register in docs table (optional)
    if session_factory:
        async with session_factory() as session:
            for doc in docs:
                checksum = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
                await _upsert_doc_orm(
                    session,
                    doc,
                    checksum=checksum,
                    source_type=FileSourceType.DOC,
                )
            await session.commit()

    # step 2: chunk all docs
    logger.info(f"Chunking {len(docs)} documents")
    chunks_per_doc = await ChunkerService.chunk_documents(docs, chunker_config)

    # step 3: flatten
    all_chunks: list[Chunk] = []
    for doc_chunks in chunks_per_doc:
        all_chunks.extend(doc_chunks)

    logger.info(f"Chunked {len(docs)} docs into {len(all_chunks)} chunks")

    # step 4: embed all chunks
    embedder = EmbedderService()
    await embedder.embed_chunks(all_chunks, embedder_config)

    # step 5: upsert to single collection
    chunks_upserted = 0
    if all_chunks:
        points = await VectorService.upsert_chunks(client, all_chunks, embedder_config)
        chunks_upserted = len(points)

    logger.info(f"Doc ingestion complete: {len(docs)} docs, {chunks_upserted} chunks")

    return {
        "docs_processed": len(docs),
        "chunks_upserted": chunks_upserted,
    }


@broker.task(max_execution_time=120)
async def ingest_single_doc(
    doc: Doc,
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
) -> AsyncTaskiqTask[dict[str, Any]]:
    """Ingest a single document into Qdrant."""
    return ingest_docs.kiq(
        [doc],
        chunker_config,
        embedder_config,
        vector_config,
        None,
    )


@broker.task(max_execution_time=600)
async def reindex_docs(
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, Any]:
    """Re-index all docs: drop collections + rebuild from PostgreSQL.

    Reads all docs from PostgreSQL where source_type='doc',
    re-chunks, re-embeds, and upserts to fresh collections.
    """
    from backend.src.api.lifecycle import Infrastructure

    infra = get_infra()
    client = infra.qdrant_client

    if not isinstance(client, AsyncQdrantClient):
        raise RuntimeError("AsyncQdrantClient required for reindex")

    # drop existing collection
    await VectorService.delete_collection(client, CHUNKS_COLLECTION)

    # recreate
    await VectorService.ensure_collection(client, embedder_config)

    # load all docs from postgres
    async with session_factory() as session:
        result = await session.execute(
            select(DocORM).where(DocORM.source_type == FileSourceType.DOC)
        )
        doc_orms = result.scalars().all()

        if not doc_orms:
            logger.info("No docs found in database to reindex")
            return {"docs_processed": 0, "chunks_upserted": 0}

        docs = [_doc_orm_to_domain(d) for d in doc_orms]

    # ingest directly (await the result)
    return await ingest_docs(
        docs,
        chunker_config,
        embedder_config,
        vector_config,
        None,
    )


async def _upsert_doc_orm(
    session: AsyncSession,
    doc: Doc,
    checksum: str,
    source_type: FileSourceType,
) -> DocORM:
    """Upsert a doc to PostgreSQL."""
    existing = await session.execute(
        select(DocORM).where(DocORM.path == doc.metadata.source)
    )
    existing_doc = existing.scalar_one_or_none()

    if existing_doc:
        existing_doc.checksum = checksum
        existing_doc.source_type = source_type.value
        return existing_doc

    doc_hash = hashlib.sha256(f"{doc.id}:{doc.metadata.source}".encode()).hexdigest()

    new_doc = DocORM(
        doc_id=doc.id,
        doc_hash=doc_hash,
        name=doc.metadata.doc_title or doc.metadata.source.split("/")[-1],
        type=SourceType.FILESYSTEM,
        path=doc.metadata.path or doc.metadata.source,
        language=doc.metadata.language,
        source_type=source_type.value,
        checksum=checksum,
        doc_summary="",
        summary_embedding="",
    )
    session.add(new_doc)
    return new_doc


def _doc_orm_to_domain(doc_orm: DocORM) -> Doc:
    """Convert DocORM to Doc domain object.

    Note: DocORM is a file registry and does NOT store content.
    For reindex, content must be loaded from file path separately.
    This function creates a stub Doc for orchestration only.
    """
    from backend.src.domain.schemas.doc import DocMetadata

    return Doc(
        id=doc_orm.doc_id,
        page_content="",
        metadata=DocMetadata(
            source=doc_orm.path or "",
            mime_type="text/plain",
            doc_id=doc_orm.doc_id,
            doc_title=doc_orm.name or "",
            path=doc_orm.path,
            language=doc_orm.language,
            source_type=doc_orm.source_type,
            checksum=doc_orm.checksum,
        ),
    )
