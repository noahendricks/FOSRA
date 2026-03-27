# %% Cell 1 - Imports and Async Setup
import asyncio
import time
import numpy as np
from pathlib import Path

# Rich and Icecream for pretty printing
from rich.pretty import pprint as pp
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.traceback import install
import icecream as ic

install(show_locals=True)

console = Console()
ic.configureOutput(prefix="DEBUG | ", includeContext=True)

pp("All imports successful!")

# %% Cell 2 - Initialize Services and Clients
embedder_config = EmbedderConfig()
vector_store_config = VectorStoreConfig(
    preferred_store=VectorStoreType.QDRANT,
    qdrant_config=QdrantConfig(
        url="http://localhost:6333",
        collection_name="codebase_chunks",
    ),
)
chunker_config = ChunkerConfig()
reranker_config = RerankerConfig()

qdrant_client = QdrantClient(url="http://localhost:6333")
qdrant_async_client = AsyncQdrantClient(url="http://localhost:6333")
falkordb_client = FalkorDB(host="localhost", port=6379)

graph_service = GraphService(client=falkordb_client, graph_name="codebase")
graph_retriever = GraphRetriever(graph_service=graph_service)

pp("=== SERVICES INITIALIZED ===")
ic(embedder_config, vector_store_config, reranker_config)
ic(type(qdrant_client), type(falkordb_client))


# %% Cell 3 - Ensure Qdrant Collections Exist
async def setup_collections():
    await VectorService.ensure_dual_collections(qdrant_async_client, embedder_config)


asyncio.run(setup_collections())
pp("Qdrant collections ensured")


# %% Cell 4 - Ensure FalkorDB Indexes Exist
def setup_graph_indexes():
    graph_service.create_indexes()


setup_graph_indexes()
pp("FalkorDB indexes created/verified")

# %% Cell 5 - INGESTION: Qdrant Docs Folder
QDRANT_DOCS_FOLDER = "/home/roccoluxe/Documents/docs/03-databases/qdrant"


async def ingest_qdrant_docs_folder():
    loader = LoaderService()
    structurer = HiChunkStructurer(config=chunker_config)
    embedder = EmbedderService()

    docs = loader._parse_directory(QDRANT_DOCS_FOLDER)
    ic(docs)

    all_chunks = []
    for doc in docs:
        chunks = HiChunk.index(document=doc, structurer=structurer)
        all_chunks.extend(chunks)

    parent_chunks = []
    leaf_chunks = []
    for chunk in all_chunks:
        level = getattr(chunk.metadata, "level", 3)
        if hasattr(chunk.metadata, "parent") and chunk.metadata.parent:
            level = chunk.metadata.parent.level
        if level in (1, 2):
            parent_chunks.append(chunk)
        else:
            leaf_chunks.append(chunk)

    if parent_chunks:
        await embedder.embed_chunks(parent_chunks, embedder_config)
    if leaf_chunks:
        await embedder.embed_chunks(leaf_chunks, embedder_config)

    parent_points = []
    leaf_points = []
    if parent_chunks:
        parent_points = await VectorService.upsert_parents(
            qdrant_async_client, parent_chunks, embedder_config
        )
    if leaf_chunks:
        leaf_points = await VectorService.upsert_chunks(
            qdrant_async_client, leaf_chunks, embedder_config
        )

    qdrant_docs_result = {
        "docs_count": len(docs),
        "parent_chunks_upserted": len(parent_points),
        "leaf_chunks_upserted": len(leaf_points),
        "docs": docs,
        "parent_chunks": parent_chunks,
        "leaf_chunks": leaf_chunks,
    }
    return qdrant_docs_result


qdrant_docs_result = asyncio.run(ingest_qdrant_docs_folder())
QDRANT_DOC_IDS = {doc.id for doc in qdrant_docs_result["docs"]}
pp("=== QDRANT DOCS INGESTION COMPLETE ===")
ic(
    qdrant_docs_result["docs_count"],
    qdrant_docs_result["parent_chunks_upserted"],
    qdrant_docs_result["leaf_chunks_upserted"],
)
ic(QDRANT_DOC_IDS)

# %% Cell 6 - INGESTION: Single Markdown File
SINGLE_FILE_PATH = "/home/roccoluxe/Documents/docs/03-databases/qdrant/01-core-concepts/documentation_concepts_filtering.md"


async def ingest_single_file():
    loader = LoaderService()
    structurer = HiChunkStructurer(config=chunker_config)
    embedder = EmbedderService()

    docs = loader._parse_files([SINGLE_FILE_PATH])

    chunks = HiChunk.index(document=docs[0], structurer=structurer)

    parent_chunks = []
    leaf_chunks = []
    for chunk in chunks:
        level = getattr(chunk.metadata, "level", 3)
        if hasattr(chunk.metadata, "parent") and chunk.metadata.parent:
            level = chunk.metadata.parent.level
        if level in (1, 2):
            parent_chunks.append(chunk)
        else:
            leaf_chunks.append(chunk)

    if parent_chunks:
        await embedder.embed_chunks(parent_chunks, embedder_config)
    if leaf_chunks:
        await embedder.embed_chunks(leaf_chunks, embedder_config)

    parent_points = []
    leaf_points = []
    if parent_chunks:
        parent_points = await VectorService.upsert_parents(
            qdrant_async_client, parent_chunks, embedder_config
        )
    if leaf_chunks:
        leaf_points = await VectorService.upsert_chunks(
            qdrant_async_client, leaf_chunks, embedder_config
        )

    single_file_result = {
        "doc": docs[0],
        "parent_chunks_upserted": len(parent_points),
        "leaf_chunks_upserted": len(leaf_points),
        "parent_chunks": parent_chunks,
        "leaf_chunks": leaf_chunks,
    }
    return single_file_result


single_file_result = asyncio.run(ingest_single_file())
SINGLE_FILE_DOC_ID = single_file_result["doc"].id
pp("=== SINGLE FILE INGESTION COMPLETE ===")
ic(single_file_result["doc"].metadata.doc_title, SINGLE_FILE_DOC_ID)
ic(
    single_file_result["parent_chunks_upserted"],
    single_file_result["leaf_chunks_upserted"],
)

# %% Cell 7 - INGESTION: Codebase (Trustgraph Monorepo)
CODEBASE_PATH = "/home/roccoluxe/trustgraph"
CODEBASE_REPO_NAME = "trustgraph"


async def ingest_codebase():
    from pathlib import Path
    import hashlib

    callgraph_service = CallGraphService()

    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }

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

    directory = Path(CODEBASE_PATH)
    files = []
    for ext in LANGUAGE_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    files = [f for f in files if not any(part in excluded_dirs for part in f.parts)]
    files = sorted(files)[:50]

    stats = {"files_processed": 0, "total_nodes": 0, "total_call_edges": 0}
    codebase_graph_results = []

    for file_path in files:
        try:
            language = LANGUAGE_EXTENSIONS.get(file_path.suffix)
            if not language:
                continue

            source_code = file_path.read_text()
            relative_path = (
                file_path.relative_to(directory)
                if file_path.is_relative_to(directory)
                else str(file_path)
            )

            file_id = hashlib.sha256(relative_path.encode()).hexdigest()[:16]

            graph_result = callgraph_service.extract_graph(
                source_code=source_code,
                file_path=relative_path,
                file_id=file_id,
                language=language,
            )

            if graph_result.nodes:
                await graph_service.upsert_file_graph(
                    graph_result=graph_result,
                    embedder_config=embedder_config,
                )
                stats["files_processed"] += 1
                stats["total_nodes"] += len(graph_result.nodes)
                stats["total_call_edges"] += len(graph_result.call_edges)
                codebase_graph_results.append(graph_result)

        except Exception as e:
            ic(file_path, e)

    codebase_result = {
        "files_processed": stats["files_processed"],
        "total_nodes": stats["total_nodes"],
        "total_call_edges": stats["total_call_edges"],
        "graph_results": codebase_graph_results,
    }
    return codebase_result


codebase_result = asyncio.run(ingest_codebase())
CODEBASE_FILE_IDS = {gr.file_id for gr in codebase_result["graph_results"]}
pp("=== CODEBASE INGESTION COMPLETE ===")
ic(codebase_result)
ic(list(CODEBASE_FILE_IDS)[:5])


# %% Cell 8 - Get Ingestion Summary Stats
async def get_ingestion_stats():
    parents_count = await VectorService.count_points(qdrant_async_client, "parents")
    chunks_count = await VectorService.count_points(qdrant_async_client, "chunks")
    return {
        "parents_collection": parents_count,
        "chunks_collection": chunks_count,
        "qdrant_docs_ingested": qdrant_docs_result["docs_count"],
        "single_file_ingested": 1,
        "codebase_files_ingested": codebase_result["files_processed"],
        "codebase_nodes": codebase_result["total_nodes"],
    }


ingestion_stats = asyncio.run(get_ingestion_stats())
pp("=== INGESTION SUMMARY ===")
ic(ingestion_stats)


# %% Cell 9 - VECTOR SEARCH: Search Qdrant Documents
async def search_qdrant_docs(query: str, limit: int = 10):
    results = await VectorService.search_collection(
        client=qdrant_client,
        collection_name="chunks",
        embed_config=embedder_config,
        query=query,
        limit=limit,
    )
    return results


QDRANT_SEARCH_QUERY = "filtering in qdrant"
qdrant_search_results = asyncio.run(search_qdrant_docs(QDRANT_SEARCH_QUERY, limit=5))
pp(f"=== QDRANT SEARCH: '{QDRANT_SEARCH_QUERY}' ===")

table = Table(title="Search Results")
table.add_column("Rank", style="cyan")
table.add_column("Score", style="magenta")
table.add_column("Doc ID", style="green")
table.add_column("Text Preview", style="white")

for i, r in enumerate(qdrant_search_results):
    table.add_row(
        str(i + 1), f"{r.score:.4f}", r.payload.get("doc_id", "N/A")[:30], r.text[:80]
    )
console.print(table)


# %% Cell 10 - VECTOR SEARCH: Dual Retrieve with Auto-Merge
async def dual_retrieve_example(query: str):
    parent_results, file_ids, merged_context = await VectorService.dual_retrieve(
        client=qdrant_client,
        embed_config=embedder_config,
        query=query,
        filters=None,
        parents_top_k=20,
        chunks_top_k=10,
        token_budget=4096,
        merge_threshold=0.5,
    )
    return {
        "parent_results": parent_results,
        "file_ids": file_ids,
        "merged_context": merged_context,
    }


DUAL_QUERY = "vector search filtering"
dual_result = asyncio.run(dual_retrieve_example(DUAL_QUERY))
pp(f"=== DUAL RETRIEVE: '{DUAL_QUERY}' ===")
ic(dual_result["parent_results"][:3] if dual_result["parent_results"] else [])
ic(dual_result["file_ids"])
pp("Merged Context Preview:")
console.print(
    Syntax(
        dual_result["merged_context"][:500],
        "markdown",
        theme="monokai",
        line_numbers=True,
    )
)


# %% Cell 11 - VECTOR SEARCH: Filtered by Doc IDs
async def search_with_doc_filter(query: str, doc_ids: set[str], limit: int = 10):
    filters = {"doc_ids": list(doc_ids)}
    results = await VectorService.search_collection(
        client=qdrant_client,
        collection_name="chunks",
        embed_config=embedder_config,
        query=query,
        filters=filters,
        limit=limit,
    )
    return results


filtered_results = asyncio.run(
    search_with_doc_filter("filter", {SINGLE_FILE_DOC_ID}, limit=5)
)
pp(f"=== FILTERED SEARCH IN SINGLE FILE ===")
ic(SINGLE_FILE_DOC_ID, len(filtered_results))
for i, r in enumerate(filtered_results):
    ic(rank=i + 1, score=r.score, text=r.text[:60])


# %% Cell 12 - RERANKER: Rerank Search Results
def test_reranker(query: str, chunks: list[RetrievedChunk]):
    reranker = RerankerService(config=reranker_config)
    reranked = reranker.rerank(query=query, chunks=chunks, top_k=5, score_threshold=0.3)
    return reranked


RERANK_QUERY = "qdrant filtering concepts"
reranked_results = test_reranker(RERANK_QUERY, qdrant_search_results)
pp(f"=== RERANKED RESULTS: '{RERANK_QUERY}' ===")
for i, chunk in enumerate(reranked_results):
    ic(i + 1, chunk.score, chunk.text[:70])


# %% Cell 13 - AUTO-MERGE: Standalone Test
def test_auto_merge(chunks: list[RetrievedChunk]):
    merged = VectorService.auto_merge(
        results=chunks, token_budget=4096, merge_threshold=0.5
    )
    return merged


merged_text = test_auto_merge(qdrant_search_results)
pp("=== AUTO-MERGE RESULT ===")
ic(len(merged_text))
console.print(Syntax(merged_text[:500], "markdown", theme="monokai", line_numbers=True))


# %% Cell 14 - GRAPH: Search by Name (Full-Text)
def test_search_by_name(
    name_pattern: str, node_type: str = "Function", limit: int = 20
):
    result = graph_retriever.search_by_name(
        name=name_pattern, node_type=node_type, limit=limit
    )
    return result


NAME_QUERY = "filter"
name_results = test_search_by_name(NAME_QUERY, limit=10)
pp(f"=== GRAPH SEARCH BY NAME: '{NAME_QUERY}' ===")
ic(name_results.total_count)
for node in name_results.nodes[:5]:
    ic(node.node_type.value, node.name, node.qualified_name, node.file_path)


# %% Cell 15 - GRAPH: Get Callers
def test_get_callers(function_name: str, depth: int = 1, limit: int = 50):
    result = graph_retriever.get_callers(name=function_name, depth=depth, limit=limit)
    return result


CALLERS_QUERY = "process"
callers_result = test_get_callers(CALLERS_QUERY, depth=1, limit=20)
pp(f"=== GET CALLERS: '{CALLERS_QUERY}' ===")
ic(callers_result.total_count)
for node in callers_result.nodes[:10]:
    ic(node.name, node.qualified_name)


# %% Cell 16 - GRAPH: Get Callees
def test_get_callees(function_name: str, limit: int = 50):
    result = graph_retriever.get_callees(name=function_name, limit=limit)
    return result


CALLEES_QUERY = "main"
callees_result = test_get_callees(CALLEES_QUERY, limit=20)
pp(f"=== GET CALLEES: '{CALLEES_QUERY}' ===")
ic(callees_result.total_count)
for node in callees_result.nodes[:10]:
    ic(node.name, node.qualified_name)


# %% Cell 17 - GRAPH: Get Call Chain
def test_get_call_chain(function_name: str, depth: int = 5, limit: int = 50):
    result = graph_retriever.get_call_chain(
        name=function_name, depth=depth, limit=limit
    )
    return result


CHAIN_QUERY = "run"
chain_result = test_get_call_chain(CHAIN_QUERY, depth=3, limit=20)
pp(f"=== GET CALL CHAIN: '{CHAIN_QUERY}' ===")
ic(chain_result.total_count)
for i, path in enumerate(chain_result.paths[:3]):
    ic(i + 1, " -> ".join([n.name for n in path]))


# %% Cell 18 - GRAPH: Get File Symbols
def test_get_file_symbols(file_id: str, limit: int = 100):
    result = graph_retriever.get_file_symbols(file_id=file_id, limit=limit)
    return result


if CODEBASE_FILE_IDS:
    SAMPLE_FILE_ID = list(CODEBASE_FILE_IDS)[0]
    symbols_result = test_get_file_symbols(SAMPLE_FILE_ID, limit=50)
    pp(f"=== FILE SYMBOLS: {SAMPLE_FILE_ID} ===")
    ic(symbols_result.total_count)
    for node in symbols_result.nodes[:10]:
        ic(node.node_type.value, node.name)


# %% Cell 19 - GRAPH: Get Inheritance Chain
def test_get_inheritance_chain(class_name: str, depth: int = 10, limit: int = 50):
    result = graph_retriever.get_inheritance_chain(
        name=class_name, depth=depth, limit=limit
    )
    return result


INHERITANCE_QUERY = "Base"
inheritance_result = test_get_inheritance_chain(INHERITANCE_QUERY, limit=20)
pp(f"=== INHERITANCE CHAIN: '{INHERITANCE_QUERY}' ===")
ic(inheritance_result.total_count)
for i, path in enumerate(inheritance_result.paths[:3]):
    ic(i + 1, " -> ".join([n.name for n in path]))


# %% Cell 20 - GRAPH: Get File Imports
def test_get_file_imports(file_id: str, limit: int = 100):
    result = graph_retriever.get_file_imports(file_id=file_id, limit=limit)
    return result


if CODEBASE_FILE_IDS:
    SAMPLE_FILE_ID = list(CODEBASE_FILE_IDS)[0]
    imports_result = test_get_file_imports(SAMPLE_FILE_ID, limit=20)
    pp(f"=== FILE IMPORTS: {SAMPLE_FILE_ID} ===")
    ic(imports_result.total_count)
    for node in imports_result.nodes[:10]:
        ic(node.name, node.file_path)


# %% Cell 21 - GRAPH: Semantic Search (requires embeddings)
async def test_semantic_search(
    query_embedding: list[float],
    node_types: list[GraphNodeType] | None = None,
    limit: int = 10,
):
    result = await graph_retriever.semantic_search(
        query_embedding=query_embedding,
        node_types=node_types,
        file_ids=None,
        limit=limit,
    )
    return result


SEMANTIC_QUERY_EMBEDDING = np.random.rand(embedder_config.dense_dimensions).tolist()
semantic_result = asyncio.run(test_semantic_search(SEMANTIC_QUERY_EMBEDDING, limit=10))
pp("=== GRAPH SEMANTIC SEARCH ===")
ic(semantic_result.total_count)
for node in semantic_result.nodes[:5]:
    ic(node.node_type.value, node.name, node.qualified_name)


# %% Cell 22 - FULL PIPELINE: Vector Search + Rerank + Auto-Merge
async def full_pipeline(query: str, top_k: int = 10):
    search_results = await VectorService.search_collection(
        client=qdrant_client,
        collection_name="chunks",
        embed_config=embedder_config,
        query=query,
        limit=top_k * 2,
    )

    if not search_results:
        return {"search_results": [], "reranked": [], "merged_context": ""}

    reranked = test_reranker(query, search_results)
    merged_context = test_auto_merge(reranked)

    return {
        "search_results": search_results,
        "reranked": reranked,
        "merged_context": merged_context,
    }


FULL_PIPELINE_QUERY = "qdrant vector filtering"
full_pipeline_result = asyncio.run(full_pipeline(FULL_PIPELINE_QUERY))
pp(f"=== FULL PIPELINE: '{FULL_PIPELINE_QUERY}' ===")
ic(
    len(full_pipeline_result["search_results"]),
    len(full_pipeline_result["reranked"]),
    len(full_pipeline_result["merged_context"]),
)
pp("Merged Context Preview:")
console.print(
    Syntax(
        full_pipeline_result["merged_context"][:400],
        "markdown",
        theme="monokai",
        line_numbers=True,
    )
)


# %% Cell 23 - UTILITY: Count All Collections
async def count_all_collections():
    parents = await VectorService.count_points(qdrant_async_client, "parents")
    chunks = await VectorService.count_points(qdrant_async_client, "chunks")
    return {"parents": parents, "chunks": chunks}


collection_counts = asyncio.run(count_all_collections())
pp("=== COLLECTION COUNTS ===")
ic(collection_counts)


# %% Cell 24 - UTILITY: Benchmark Search Latency
async def benchmark_search(queries: list[str]):
    results = []
    for query in queries:
        start = time.time()
        search_results = await VectorService.search_collection(
            client=qdrant_client,
            collection_name="chunks",
            embed_config=embedder_config,
            query=query,
            limit=10,
        )
        elapsed = (time.time() - start) * 1000
        results.append(
            {"query": query, "results": len(search_results), "latency_ms": elapsed}
        )
    return results


BENCHMARK_QUERIES = ["filtering vectors", "qdrant collections", "semantic search"]
benchmark_results = asyncio.run(benchmark_search(BENCHMARK_QUERIES))
pp("=== SEARCH LATENCY BENCHMARK ===")
ic(benchmark_results)

pp("=== REPL Ready for Interactive Testing ===")
pp(
    {
        "available_variables": [
            "qdrant_docs_result",
            "single_file_result",
            "codebase_result",
            "ingestion_stats",
            "qdrant_search_results",
            "reranked_results",
            "dual_result",
            "name_results",
            "callers_result",
            "callees_result",
            "chain_result",
            "semantic_result",
        ]
    }
)
