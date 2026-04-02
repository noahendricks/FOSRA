"""Ingestion API endpoints for indexing codebases and documents.

Provides REST endpoints for:
- Indexing codebases into FalkorDB (code graph)
- Indexing documents into Qdrant (vector store)
- Checking ingestion status
- Re-indexing (drop + rebuild)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from loguru import logger

from backend.src.api.dependencies import get_db_session
from backend.src.api.lifecycle import global_infra
from backend.src.domain.enums import FileSourceType, VectorStoreType
from backend.src.domain.schemas.doc import Doc, DocMetadata
from backend.src.settings import (
    ChunkerConfig,
    EmbedderConfig,
    VectorStoreConfig,
    settings,
)
from backend.src.storage.models import ulid_factory

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class CodebaseIngestRequest:
    directory_path: str
    repo_name: str | None = None
    language_filter: list[str] | None = None
    force: bool = False


class DocIngestRequest:
    file_paths: list[str]
    source_type: str = "doc"
    force: bool = False


class IngestStatusResponse:
    postgres_files: int
    qdrant_parents: int
    qdrant_chunks: int
    falkordb_nodes: int
    falkordb_edges: int


@router.post("/codebase")
async def ingest_codebase(
    directory_path: Annotated[str, Body()],
    repo_name: Annotated[str | None, Body()] = None,
    language_filter: Annotated[list[str] | None, Body()] = None,
    force: Annotated[bool, Body()] = False,
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Ingest a codebase directory into FalkorDB.

    Args:
        directory_path: Absolute path to the codebase directory
        repo_name: Optional repository name (defaults to directory name)
        language_filter: Optional list of languages to index (e.g., ["python", "go"])
        force: If True, re-index even if checksum unchanged

    Returns:
        Summary of indexed files, nodes, and edges
    """
    from backend.src.tasks.codebase_ingestion import ingest_codebase as _ingest

    path = Path(directory_path)
    if not path.exists():
        raise HTTPException(
            status_code=400, detail=f"Directory not found: {directory_path}"
        )

    if not path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Not a directory: {directory_path}"
        )

    repo = repo_name or path.name

    if global_infra.falkordb_client is None:
        raise HTTPException(status_code=503, detail="FalkorDB not available")
    if global_infra.session_factory is None:
        raise HTTPException(
            status_code=503, detail="Database session factory not available"
        )

    try:
        embedder_config = _default_embedder_config()
        result = await _ingest(
            directory_path=str(path.absolute()),
            repo_name=repo,
            embedder_config=embedder_config,
            falkordb_client=global_infra.falkordb_client,
            session_factory=global_infra.session_factory,
            recursive=True,
        )

        logger.info(
            "Codebase ingestion complete: {} files, {} nodes, {} edges",
            result.get("files_indexed", 0),
            result.get("nodes_created", 0),
            result.get("edges_created", 0),
        )

        return result

    except Exception as e:
        logger.error("Codebase ingestion failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/codebase/file")
async def ingest_single_file(
    file_path: Annotated[str, Body()],
    repo_name: Annotated[str, Body()] = "",
    force: Annotated[bool, Body()] = False,
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Ingest a single code file into FalkorDB.

    Args:
        file_path: Absolute path to the code file
        repo_name: Repository name for grouping
        force: If True, re-index even if checksum unchanged

    Returns:
        Summary of nodes and edges created
    """
    from backend.src.tasks.codebase_ingestion import ingest_single_file as _ingest

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    if global_infra.falkordb_client is None:
        raise HTTPException(status_code=503, detail="FalkorDB not available")
    if global_infra.session_factory is None:
        raise HTTPException(
            status_code=503, detail="Database session factory not available"
        )

    try:
        embedder_config = _default_embedder_config()
        result = await _ingest(
            file_path=str(path.absolute()),
            repo_name=repo_name,
            embedder_config=embedder_config,
            falkordb_client=global_infra.falkordb_client,
            session_factory=global_infra.session_factory,
        )

        return result

    except Exception as e:
        logger.error("Single file ingestion failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docs")
async def ingest_documents(
    file_paths: Annotated[list[str], Body()],
    source_type: Annotated[str, Body()] = "doc",
    force: Annotated[bool, Body()] = False,
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Ingest documents into Qdrant (dual collections: parents + chunks).

    Args:
        file_paths: List of absolute paths to documents
        source_type: "doc" for regular docs, "code-in-doc" for code documentation
        force: If True, re-index even if checksum unchanged

    Returns:
        Summary of documents and chunks indexed
    """
    from backend.src.tasks.doc_ingestion import ingest_docs as _ingest

    paths = [Path(p) for p in file_paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Files not found: {missing}")

    docs: list[Doc] = []
    for p in paths:
        from backend.src.services.processing.docling_loader import (
            DoclingLoader,
        )

        mime = _guess_mime(p)
        doc = DoclingLoader.parse_file_sync(str(p.absolute()), mime_type=mime)
        doc.metadata.source_type = (
            FileSourceType.DOC if source_type == "doc" else FileSourceType.CODEBASE
        )
        docs.append(doc)

    embedder_config = _default_embedder_config()
    chunker_config = ChunkerConfig()
    vector_config = _default_vector_config()

    try:
        result = await _ingest(
            docs=docs,
            chunker_config=chunker_config,
            embedder_config=embedder_config,
            vector_config=vector_config,
            session_factory=global_infra.session_factory,
        )

        logger.info(
            "Document ingestion complete: {} docs, {} parent chunks, {} child chunks",
            result.get("docs_indexed", 0),
            result.get("parent_chunks", 0),
            result.get("child_chunks", 0),
        )

        return result

    except Exception as e:
        logger.error("Document ingestion failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_ingestion_status(
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Get ingestion status across all stores.

    Returns:
        Counts from PostgreSQL, Qdrant, and FalkorDB
    """
    from sqlalchemy import func, select

    from backend.src.services.retrieval.vector_service import (
        CHUNKS_COLLECTION,
        VectorService,
    )
    from backend.src.storage.models import DocORM

    postgres_count = await session.scalar(select(func.count(DocORM.doc_id)))

    qdrant_client = global_infra.qdrant_client
    qdrant_parents = 0
    qdrant_chunks = 0

    if qdrant_client:
        try:
            embedder_config = _default_embedder_config()
            qdrant_parents = await VectorService.count_points(
                qdrant_client, CHUNKS_COLLECTION
            )
            qdrant_chunks = await VectorService.count_points(
                qdrant_client, CHUNKS_COLLECTION
            )
        except Exception as e:
            logger.warning("Could not get Qdrant counts: {}", e)

    falkordb_nodes = 0
    falkordb_edges = 0

    if global_infra.falkordb_graph:
        try:
            graph = global_infra.falkordb_graph
            node_result = graph.query("MATCH (n) RETURN count(n) as count")
            if node_result.result_set:
                falkordb_nodes = node_result.result_set[0][0]

            edge_result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            if edge_result.result_set:
                falkordb_edges = edge_result.result_set[0][0]
        except Exception as e:
            logger.warning("Could not get FalkorDB counts: {}", e)

    return {
        "postgres_files": postgres_count or 0,
        "qdrant_parents": qdrant_parents,
        "qdrant_chunks": qdrant_chunks,
        "falkordb_nodes": falkordb_nodes,
        "falkordb_edges": falkordb_edges,
    }


@router.delete("/codebase")
async def reindex_codebase(
    directory_path: Annotated[str, Query()],
    repo_name: Annotated[str, Query()],
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Re-index a codebase (drop existing + rebuild).

    Args:
        directory_path: Absolute path to the codebase directory
        repo_name: Repository name to re-index

    Returns:
        Summary of re-indexing
    """
    from backend.src.tasks.codebase_ingestion import reindex_codebase as _reindex

    if global_infra.falkordb_client is None:
        raise HTTPException(status_code=503, detail="FalkorDB not available")
    if global_infra.session_factory is None:
        raise HTTPException(
            status_code=503, detail="Database session factory not available"
        )

    embedder_config = _default_embedder_config()
    try:
        result = await _reindex(
            directory_path=directory_path,
            repo_name=repo_name,
            embedder_config=embedder_config,
            falkordb_client=global_infra.falkordb_client,
            session_factory=global_infra.session_factory,
        )

        return result

    except Exception as e:
        logger.error("Codebase re-index failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/docs")
async def reindex_docs(
    collection: Annotated[str, Query()] = "all",
    session=Depends(get_db_session),
) -> dict[str, Any]:
    """Re-index documents (drop collections + rebuild from PostgreSQL).

    Args:
        collection: "all", "parents", or "chunks"

    Returns:
        Summary of re-indexing
    """
    from backend.src.tasks.doc_ingestion import reindex_docs as _reindex

    if global_infra.session_factory is None:
        raise HTTPException(
            status_code=503, detail="Database session factory not available"
        )

    embedder_config = _default_embedder_config()
    chunker_config = ChunkerConfig()
    vector_config = _default_vector_config()
    try:
        result = await _reindex(
            chunker_config,
            embedder_config,
            vector_config,
            global_infra.session_factory,
        )
        return result

    except Exception as e:
        logger.error("Document re-index failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


def _guess_mime(path: Path) -> str:
    """Guess MIME type from file extension."""
    suffix = path.suffix.lower()

    type_map = {
        ".md": "text/markdown",
        ".mdx": "text/markdown",
        ".txt": "text/plain",
        ".rst": "text/x-rst",
        ".html": "text/html",
        ".pdf": "application/pdf",
        ".py": "text/x-python",
        ".go": "text/x-go",
        ".js": "text/javascript",
        ".ts": "text/typescript",
        ".tsx": "text/typescript-jsx",
        ".rs": "text/x-rust",
        ".java": "text/x-java",
        ".cpp": "text/x-c++",
        ".c": "text/x-c",
        ".h": "text/x-c",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".toml": "text/x-toml",
    }

    return type_map.get(suffix, "application/octet-stream")


def _default_embedder_config() -> EmbedderConfig:
    """Get default embedder config from settings."""
    return EmbedderConfig(
        embedder_type=settings.embedding.model_type,
        dense_model=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )


def _default_vector_config() -> VectorStoreConfig:
    """Get default vector store config from settings."""
    from backend.src.settings import QdrantConfig

    store_type = VectorStoreType(settings.vectors.vector_store_type)
    return VectorStoreConfig(
        preferred_store=store_type,
        qdrant_config=QdrantConfig(
            collection_name=settings.vectors.collection_name,
            host=settings.qdrant.host,
            port=settings.qdrant.port,
        ),
    )
