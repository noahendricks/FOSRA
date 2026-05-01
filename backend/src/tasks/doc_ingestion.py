from __future__ import annotations

import hashlib
import msgspec
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from taskiq import AsyncTaskiqTask

from backend.src.domain.enums import FileSourceType
from backend.src.settings import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import Doc, Subsection
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.vector_service import (
    CHUNKS_COLLECTION,
    VectorService,
)
from backend.src.storage.models import DocORM

from .broker import broker, get_infra


async def _upsert_doc_orm(
    session: AsyncSession,
    doc: Doc,
    checksum: str,
    source_type: FileSourceType,
) -> DocORM:
    """Upsert a doc record in PostgreSQL, matching by path+repo."""
    from pathlib import Path
    from sqlalchemy import select

    # Handle both dict and DocMetadata (msgspec.Struct) metadata fields
    metadata = doc.metadata
    if isinstance(metadata, dict):
        path = metadata.get("path") or ""
        repo = metadata.get("repo")
        doc_title = metadata.get("doc_title") or ""
    else:
        path = metadata.path or ""
        repo = getattr(metadata, "repo", None)
        doc_title = getattr(metadata, "doc_title", "") or ""

    result = await session.execute(
        select(DocORM).where(
            DocORM.path == path,
            DocORM.repo == repo,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.checksum = checksum
        existing.doc_hash = checksum
        if doc_title:
            existing.name = doc_title
        return existing

    doc_orm = DocORM(
        path=path,
        name=doc_title or str(Path(path).name) if path else doc.id,
        language=metadata.get("language") if isinstance(metadata, dict) else getattr(metadata, "language", None),
        repo=repo,
        source_type=source_type.value,
        checksum=checksum,
        doc_hash=checksum,
        type="doc",
        doc_summary="",
        summary_embedding="",
    )
    session.add(doc_orm)
    return doc_orm


@broker.task(max_execution_time=300)
async def ingest_docs(
    docs: list[dict[str, Any]],
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict[str, Any]:
    """Ingest documents into Qdrant.

    Flow:
    1. Reconstruct Doc objects from dict (taskiq serialization)
    2. Register docs in PostgreSQL (optional, if session_factory provided)
    3. Chunk via HiChunk (L1/L2)
    4. Embed all chunks
    5. Upsert to single collection

    Args:
        docs: List of Doc objects as dicts (for taskiq serialization)
        chunker_config: Chunker configuration
        embedder_config: Embedder configuration
        vector_config: Vector store configuration
        session_factory: Optional async session factory for PostgreSQL registration

    Returns:
        dict with counts: {chunks_upserted, docs_processed}
    """
    # Reconstruct Doc objects from dicts using msgspec.convert
    # (plain Doc(**d) doesn't auto-convert nested dicts to structs)
    domain_docs = [msgspec.convert(d, Doc) for d in docs]

    infra = get_infra()

    # ensure collection exists
    client = infra.qdrant_client
    if client is None:
        raise RuntimeError("Qdrant client not initialized in infrastructure")
    await VectorService.ensure_collection(client, embedder_config)

    # step 1: register in docs table (optional)
    if session_factory:
        async with session_factory() as session:
            for doc in domain_docs:
                checksum = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
                _ = await _upsert_doc_orm(
                    session,
                    doc,
                    checksum=checksum,
                    source_type=FileSourceType.DOC,
                )
            await session.commit()

    # step 2: chunk all docs
    logger.info("Chunking {} documents", len(domain_docs))
    chunks_per_doc = await ChunkerService.chunk_documents(domain_docs, chunker_config)

    # step 3: flatten
    all_chunks: list[Subsection] = []
    for doc_chunks in chunks_per_doc:
        all_chunks.extend(doc_chunks)

    logger.info("Chunked {} docs into {} chunks", len(domain_docs), len(all_chunks))

    # step 4: embed all chunks
    embedder = EmbedderService()
    _ = await embedder.embed_chunks(all_chunks, embedder_config)

    # step 5: upsert to single collection
    chunks_upserted = 0
    if all_chunks:
        points = await VectorService.upsert_chunks(client, all_chunks, embedder_config)
        chunks_upserted = len(points)

    logger.bind(_structured={"docs": len(domain_docs), "chunks": chunks_upserted}).info(
        "Doc ingestion complete"
    )

    return {
        "docs_processed": len(domain_docs),
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
    return await ingest_docs.kiq(
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
    infra = get_infra()

    # drop existing collection
    client = infra.qdrant_client
    if client is None:
        raise RuntimeError("Qdrant client not initialized in infrastructure")
    _ = await VectorService.delete_collection(client, CHUNKS_COLLECTION)

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

        docs = []
        for doc_orm in doc_orms:
            # Load content from file path
            try:
                with open(doc_orm.path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(
                    "Failed to read file {} for reindexing: {}", doc_orm.path, e
                )
                continue

            docs.append(_doc_orm_to_domain_with_content(doc_orm, content))

        # ingest directly (await the result) - pass dicts for taskiq serialization
        return await ingest_docs(
            [doc.to_dict() for doc in docs],
            chunker_config,
            embedder_config,
            vector_config,
            None,
        )


async def _doc_orm_to_domain_with_content(doc_orm: DocORM, content: str) -> Doc:
    """Convert DocORM to Doc domain object with provided content."""
    from backend.src.domain.schemas.doc import DocMetadata

    return Doc(
        id=doc_orm.doc_id,
        page_content=content,
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
