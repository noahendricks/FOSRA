from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypedDict

from falkordb import FalkorDB
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.src.domain.enums import FileSourceType
from backend.src.domain.schemas.graph import ResolvedImport
from backend.src.settings import EmbedderConfig
from backend.src.services.processing.callgraph_service import CallGraphService
from backend.src.services.retrieval.graph_service import GraphService
from backend.src.storage.models import DocORM

from .broker import broker


class IngestionStats(TypedDict):
    files_processed: int
    total_nodes: int
    total_call_edges: int
    total_inheritance_edges: int
    total_import_edges: int
    errors: list[dict[str, str]]


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


@broker.task(max_execution_time=600)
async def ingest_codebase(
    directory_path: str,
    repo_name: str | None,
    embedder_config: EmbedderConfig,
    falkordb_client: "FalkorDB",
    session_factory: async_sessionmaker[AsyncSession],
    recursive: bool = True,
) -> IngestionStats:
    """
    ingest a codebase directory into falkordb.

    - registers files in postgres docs table
    - extracts code graph (nodes, call edges, inheritance edges)
    - upserts to falkordb with embeddings
    """
    callgraph_service = CallGraphService()
    graph_service = GraphService(falkordb_client, graph_name=repo_name or "codebase")

    graph_service.create_indexes(embedder_config)

    stats: IngestionStats = {
        "files_processed": 0,
        "total_nodes": 0,
        "total_call_edges": 0,
        "total_inheritance_edges": 0,
        "total_import_edges": 0,
        "errors": [],
    }

    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")

    files = _collect_code_files(directory, recursive)
    logger.info("Found {} code files in {}", len(files), directory_path)

    all_imports: list[ResolvedImport] = []

    async with session_factory() as session:
        for file_path in files:
            try:
                file_stats, file_imports = await _process_file(
                    file_path=file_path,
                    repo_name=repo_name,
                    embedder_config=embedder_config,
                    callgraph_service=callgraph_service,
                    graph_service=graph_service,
                    session=session,
                    base_dir=directory,
                )
                all_imports.extend(file_imports)

                stats["files_processed"] += 1
                stats["total_nodes"] += file_stats.get("nodes_created", 0)
                stats["total_call_edges"] += file_stats.get("edges_created", 0)

            except Exception as e:
                logger.opt(exception=True).error("Failed to process {}", file_path)
                stats["errors"].append({"file": str(file_path), "error": str(e)})

        await session.commit()

    resolve_stats = await graph_service.resolve_imports(all_imports)
    stats["total_import_edges"] = resolve_stats.get("edges_created", 0)

    logger.bind(_structured=stats).info("Codebase ingestion complete")
    return stats  # type: ignore[return-value]


@broker.task(max_execution_time=120)
async def ingest_single_file(
    file_path: str,
    repo_name: str | None,
    embedder_config: EmbedderConfig,
    falkordb_client: "FalkorDB",
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, Any]:
    """
    ingest a single code file into falkordb.
    """
    callgraph_service = CallGraphService()
    graph_service = GraphService(falkordb_client, graph_name=repo_name or "codebase")

    graph_service.create_indexes(embedder_config)

    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    async with session_factory() as session:
        stats, _ = await _process_file(
            file_path=path,
            repo_name=repo_name,
            embedder_config=embedder_config,
            callgraph_service=callgraph_service,
            graph_service=graph_service,
            session=session,
            base_dir=path.parent,
        )
        await session.commit()

    return stats


async def _process_file(
    file_path: Path,
    repo_name: str | None,
    embedder_config: EmbedderConfig,
    callgraph_service: CallGraphService,
    graph_service: GraphService,
    session: AsyncSession,
    base_dir: Path,
) -> tuple[dict[str, Any], list[ResolvedImport]]:
    language = LANGUAGE_EXTENSIONS.get(file_path.suffix)
    if not language:
        logger.warning("Skipping unsupported file type: {}", file_path)
        return {"nodes_created": 0, "edges_created": 0}, []

    source_code = file_path.read_text()

    relative_path = (
        str(file_path.relative_to(base_dir))
        if file_path.is_relative_to(base_dir)
        else str(file_path)
    )

    checksum = hashlib.sha256(f"{relative_path}:{source_code}".encode()).hexdigest()

    existing = await session.execute(
        select(DocORM).where(
            DocORM.path == relative_path,
            DocORM.repo == repo_name,
        )
    )

    existing_doc = existing.scalar_one_or_none()

    if existing_doc and existing_doc.checksum == checksum:
        logger.debug("File unchanged, skipping: {}", relative_path)
        return {"nodes_created": 0, "edges_created": 0, "skipped": True}, []

    if existing_doc:
        file_id = existing_doc.doc_id
        existing_doc.checksum = checksum
        existing_doc.doc_hash = checksum
    else:
        doc = DocORM(
            path=relative_path,
            language=language,
            repo=repo_name,
            source_type=FileSourceType.CODEBASE.value,
            checksum=checksum,
            doc_hash=checksum,
            name=file_path.name,
            type="code",
            doc_summary="",
            summary_embedding="",
        )

        session.add(doc)
        await session.flush()
        file_id = doc.doc_id

    graph_result = callgraph_service.extract_graph(
        source_code=source_code,
        file_path=relative_path,
        file_id=file_id,
        language=language,
    )

    upsert_stats = await graph_service.upsert_file_graph(
        graph_result=graph_result,
        embedder_config=embedder_config,
    )

    logger.bind(_structured={"path": relative_path, **upsert_stats}).info(
        "Processed file"
    )
    return upsert_stats, graph_result.imports


def _collect_code_files(directory: Path, recursive: bool) -> list[Path]:
    """
    collect all code files in a directory.
    """
    files = []

    if recursive:
        for ext in LANGUAGE_EXTENSIONS:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in LANGUAGE_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))

    excluded_dirs = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        "target",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }

    files = [f for f in files if not any(part in excluded_dirs for part in f.parts)]

    return sorted(files)


@broker.task(max_execution_time=900)
async def reindex_codebase(
    directory_path: str,
    repo_name: str | None,
    embedder_config: EmbedderConfig,
    falkordb_client: "FalkorDB",
    session_factory: async_sessionmaker[AsyncSession],
) -> IngestionStats:
    """
    full re-index: clear graph and rebuild from scratch.
    """
    graph_service = GraphService(falkordb_client, graph_name=repo_name or "codebase")
    graph_service.clear_graph()

    return await ingest_codebase(
        directory_path=directory_path,
        repo_name=repo_name,
        embedder_config=embedder_config,
        falkordb_client=falkordb_client,
        session_factory=session_factory,
    )
