from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.src.domain.enums import FileSourceType, SourceType
from backend.src.domain.schemas.config import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
)
from backend.src.domain.schemas.doc import Chunk, Doc
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.retrieval.vector_service import (
    CHUNKS_COLLECTION,
    PARENTS_COLLECTION,
    VectorService,
)
from backend.src.storage.models import DocORM

from .broker import broker, get_infra


@broker.task
async def ingest_docs(
    docs: list[Doc],
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> dict:
    """Ingest documents into Qdrant with dual collections (parents + chunks).

    Flow:
    1. Register docs in PostgreSQL (optional, if session_factory provided)
    2. Chunk via HiChunk (L1/L2/L3)
    3. Separate by level
    4. Embed all chunks
    5. Upsert parents → parents collection
    6. Upsert leaf chunks → chunks collection

    Args:
        docs: List of Doc objects with page_content and metadata
        chunker_config: Chunker configuration
        embedder_config: Embedder configuration
        vector_config: Vector store configuration
        session_factory: Optional async session factory for PostgreSQL registration

    Returns:
        dict with counts: {parents_upserted, chunks_upserted, docs_processed}
    """
    from backend.src.api.lifecycle import Infrastructure

    infra = get_infra()
    client = infra.qdrant_client

    if not isinstance(client, QdrantClient):
        raise RuntimeError("QdrantClient required for doc ingestion")

    # ensure collections exist
    VectorService.ensure_dual_collections(client, embedder_config)

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

    # step 3: flatten and separate by level
    all_chunks: list[Chunk] = []
    for doc_chunks in chunks_per_doc:
        all_chunks.extend(doc_chunks)

    parent_chunks: list[Chunk] = []
    leaf_chunks: list[Chunk] = []

    for chunk in all_chunks:
        level = getattr(chunk.metadata, "level", 3)
        if hasattr(chunk.metadata, "parent") and chunk.metadata.parent:
            level = chunk.metadata.parent.level

        if level in (1, 2):
            parent_chunks.append(chunk)
        else:
            leaf_chunks.append(chunk)

    logger.info(
        f"Separated {len(parent_chunks)} parent chunks, {len(leaf_chunks)} leaf chunks"
    )

    # step 4: embed all chunks
    embedder = EmbedderService()

    if parent_chunks:
        logger.info(f"Embedding {len(parent_chunks)} parent chunks")
        await embedder.embed_chunks(parent_chunks, embedder_config)

    if leaf_chunks:
        logger.info(f"Embedding {len(leaf_chunks)} leaf chunks")
        await embedder.embed_chunks(leaf_chunks, embedder_config)

    # step 5: upsert to collections
    parents_upserted = 0
    chunks_upserted = 0

    if parent_chunks:
        points = await VectorService.upsert_parents(
            client, parent_chunks, embedder_config
        )
        parents_upserted = len(points)

    if leaf_chunks:
        points = await VectorService.upsert_chunks(client, leaf_chunks, embedder_config)
        chunks_upserted = len(points)

    logger.info(
        f"Doc ingestion complete: {len(docs)} docs, "
        f"{parents_upserted} parents, {chunks_upserted} chunks"
    )

    return {
        "docs_processed": len(docs),
        "parents_upserted": parents_upserted,
        "chunks_upserted": chunks_upserted,
    }


@broker.task
async def ingest_single_doc(
    doc: Doc,
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
) -> dict:
    """Ingest a single document into Qdrant."""
    return await ingest_docs.kiq(
        [doc],
        chunker_config,
        embedder_config,
        vector_config,
        None,
    )


@broker.task
async def reindex_docs(
    chunker_config: ChunkerConfig,
    embedder_config: EmbedderConfig,
    vector_config: VectorStoreConfig,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict:
    """Re-index all docs: drop collections + rebuild from PostgreSQL.

    Reads all docs from PostgreSQL where source_type='doc',
    re-chunks, re-embeds, and upserts to fresh collections.
    """
    from backend.src.api.lifecycle import Infrastructure

    infra = get_infra()
    client = infra.qdrant_client

    if not isinstance(client, QdrantClient):
        raise RuntimeError("QdrantClient required for reindex")

    # drop existing collections
    VectorService.delete_collection(client, PARENTS_COLLECTION)
    VectorService.delete_collection(client, CHUNKS_COLLECTION)

    # recreate
    VectorService.ensure_dual_collections(client, embedder_config)

    # load all docs from postgres
    async with session_factory() as session:
        result = await session.execute(
            select(DocORM).where(DocORM.source_type == FileSourceType.DOC)
        )
        doc_orms = result.scalars().all()

        if not doc_orms:
            logger.info("No docs found in database to reindex")
            return {"docs_processed": 0, "parents_upserted": 0, "chunks_upserted": 0}

        docs = [_doc_orm_to_domain(d) for d in doc_orms]

    # ingest
    return await ingest_docs.kiq(
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
