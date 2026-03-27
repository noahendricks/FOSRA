# %% Cell 1 - Imports
import asyncio
import time
import numpy as np
from pathlib import Path

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient

# FalkorDB imports
from falkordb import FalkorDB

# Config imports
from backend.src.settings.config import (
    EmbedderConfig,
    VectorStoreConfig,
    RerankerConfig,
    QdrantConfig,
    ChunkerConfig,
    LLMConfig,
)
from backend.src.domain.enums import VectorStoreType, GraphNodeType

# Service imports
from backend.src.services.retrieval.vector_service import VectorService, RetrievedChunk
from backend.src.services.retrieval.graph_service import GraphService
from backend.src.services.retrieval.graph_retriever import GraphRetriever
from backend.src.services.retrieval.reranker_service import RerankerService

# Processing imports
from backend.src.services.processing.chunker_service import ChunkerService
from backend.src.services.processing.embedder_service import EmbedderService
from backend.src.services.processing.loader_service import LoaderService
from backend.src.services.processing.hi_chunk import HiChunk, HiChunkStructurer
from backend.src.services.processing.callgraph_service import CallGraphService

# Conversation/Agentic imports
from backend.src.services.conversation.retrieval_pipeline import (
    build_retrieval_pipeline,
    RetrievalState,
)
from backend.src.services.conversation.query_expander import QueryExpander
from backend.src.services.conversation.subagent import Subagent
from backend.src.services.conversation.tools import (
    RetrievalResultStore,
    create_retrieval_tool,
)
from backend.src.services.conversation.agent_service import create_fosra_agent
from backend.src.services.conversation.llm_service import LLMService

# Domain imports
from backend.src.domain.schemas.doc import Doc, Chunk
from backend.src.domain.schemas.retrieval import AccumulatedContext

print("All imports successful!")

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

# LLM Config for agentic loop
llm_config = LLMService._resolve_llm_config(None)  # Uses defaults

qdrant_client = QdrantClient(url="http://localhost:6333")
qdrant_async_client = AsyncQdrantClient(url="http://localhost:6333")
falkordb_client = FalkorDB(host="localhost", port=6379)

graph_service = GraphService(client=falkordb_client, graph_name="codebase")
graph_retriever = GraphRetriever(graph_service=graph_service)

print("All services initialized!")
print(f"LLM Config: {llm_config.provider}/{llm_config.model}")


# %% Cell 3 - Ensure Qdrant Collections Exist
async def setup_collections():
    await VectorService.ensure_dual_collections(qdrant_async_client, embedder_config)
    print("Collections ensured!")


asyncio.run(setup_collections())


# %% Cell 4 - Ensure FalkorDB Indexes Exist
def setup_graph_indexes():
    graph_service.create_indexes()
    print("Graph indexes created/verified!")


setup_graph_indexes()

# %% Cell 5 - INGESTION: Qdrant Docs Folder
QDRANT_DOCS_FOLDER = "/home/roccoluxe/Documents/docs/03-databases/qdrant"


async def ingest_qdrant_docs_folder():
    """Ingest all docs from the Qdrant folder into Qdrant vector store."""
    loader = LoaderService()
    structurer = HiChunkStructurer(config=chunker_config)
    embedder = EmbedderService()

    docs = loader._parse_directory(QDRANT_DOCS_FOLDER)
    print(f"Loaded {len(docs)} documents from {QDRANT_DOCS_FOLDER}")

    all_chunks = []
    for doc in docs:
        chunks = HiChunk.index(document=doc, structurer=structurer)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")

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

    print(
        f"Separated into {len(parent_chunks)} parent chunks and {len(leaf_chunks)} leaf chunks"
    )

    if parent_chunks:
        await embedder.embed_chunks(parent_chunks, embedder_config)
    if leaf_chunks:
        await embedder.embed_chunks(leaf_chunks, embedder_config)
    print("Embeddings created")

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
    print(
        f"Qdrant docs ingestion complete: {qdrant_docs_result['docs_count']} docs, "
        f"{qdrant_docs_result['parent_chunks_upserted']} parents, "
        f"{qdrant_docs_result['leaf_chunks_upserted']} leaf chunks"
    )
    return qdrant_docs_result


qdrant_docs_result = asyncio.run(ingest_qdrant_docs_folder())
QDRANT_DOC_IDS = {doc.id for doc in qdrant_docs_result["docs"]}
print(f"Qdrant doc IDs: {QDRANT_DOC_IDS}")

# %% Cell 6 - INGESTION: Single Markdown File
SINGLE_FILE_PATH = "/home/roccoluxe/Documents/docs/03-databases/qdrant/01-core-concepts/documentation_concepts_filtering.md"


async def ingest_single_file():
    """Ingest a single markdown file into Qdrant."""
    loader = LoaderService()
    structurer = HiChunkStructurer(config=chunker_config)
    embedder = EmbedderService()

    docs = loader._parse_files([SINGLE_FILE_PATH])
    print(f"Loaded document: {docs[0].metadata.doc_title}")

    chunks = HiChunk.index(document=docs[0], structurer=structurer)
    print(f"Created {len(chunks)} chunks")

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
    print(
        f"Single file ingestion complete: {len(parent_points)} parents, {len(leaf_points)} leaf chunks"
    )
    return single_file_result


single_file_result = asyncio.run(ingest_single_file())
SINGLE_FILE_DOC_ID = single_file_result["doc"].id
print(f"Single file doc ID: {SINGLE_FILE_DOC_ID}")

# %% Cell 7 - INGESTION: Codebase (Trustgraph Monorepo)
CODEBASE_PATH = "/home/roccoluxe/trustgraph"
CODEBASE_REPO_NAME = "trustgraph"


async def ingest_codebase():
    """Ingest the trustgraph codebase into FalkorDB."""
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

    print(f"Found {len(files)} code files to ingest (limited to 50 for demo)")

    stats = {
        "files_processed": 0,
        "total_nodes": 0,
        "total_call_edges": 0,
    }

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
            print(f"Error processing {file_path}: {e}")

    codebase_result = {
        "files_processed": stats["files_processed"],
        "total_nodes": stats["total_nodes"],
        "total_call_edges": stats["total_call_edges"],
        "graph_results": codebase_graph_results,
    }

    print(
        f"Codebase ingestion complete: {stats['files_processed']} files, "
        f"{stats['total_nodes']} nodes, {stats['total_call_edges']} edges"
    )
    return codebase_result


codebase_result = asyncio.run(ingest_codebase())
CODEBASE_FILE_IDS = {gr.file_id for gr in codebase_result["graph_results"]}
print(f"Codebase file IDs (sample): {list(CODEBASE_FILE_IDS)[:10]}")


# %% Cell 8 - INGESTION SUMMARY
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
print("=== INGESTION SUMMARY ===")
for k, v in ingestion_stats.items():
    print(f"  {k}: {v}")


# %% Cell 9 - STEP 1: Query Expansion
async def test_query_expansion(user_query: str):
    """Test query expansion step."""
    expansion = await QueryExpander.expand(
        user_query=user_query,
        llm_config=llm_config,
        chat_history=None,
    )
    return expansion


QUERY = "How does filtering work in Qdrant?"
query_expansion = asyncio.run(test_query_expansion(QUERY))
print(f"=== Query Expansion: '{QUERY}' ===")
print(f"  Rewritten query: {query_expansion.rewritten_query}")
print(f"  Checklist ({len(query_expansion.checklist)} items):")
for item in query_expansion.checklist:
    status = "✓" if item.answered else "○"
    print(f"    {status} [{item.id}] {item.question}")


# %% Cell 10 - STEP 2: Initial Retrieval (Dual Retrieve)
async def test_initial_retrieval(query: str):
    """Test initial retrieval with dual_retrieve."""
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


INITIAL_QUERY = query_expansion.rewritten_query
initial_retrieval_result = asyncio.run(test_initial_retrieval(INITIAL_QUERY))
print(f"=== Initial Retrieval ===")
print(f"  Query: {INITIAL_QUERY}")
print(f"  Parent results: {len(initial_retrieval_result['parent_results'])}")
print(f"  File IDs: {initial_retrieval_result['file_ids']}")
print(f"  Merged context ({len(initial_retrieval_result['merged_context'])} chars):")
print(f"  {initial_retrieval_result['merged_context'][:300]}...")


# %% Cell 11 - STEP 3: Agentic Loop - Assess and Plan
async def test_agentic_loop(
    original_query: str,
    checklist: list,
    context: AccumulatedContext,
    iteration: int,
):
    """Test a single agentic loop iteration."""
    result = await Subagent.assess_and_plan(
        original_query=original_query,
        checklist=checklist,
        context=context,
        iteration=iteration,
        llm_config=llm_config,
        max_iterations=3,
    )
    return result


# Build initial context from merged text
initial_items = []
for chunk in initial_retrieval_result["parent_results"]:
    from backend.src.domain.schemas.retrieval import AccumulatedItem

    initial_items.append(
        AccumulatedItem(
            file_id=chunk.payload.get("doc_id", ""),
            path=chunk.payload.get("doc_title", "unknown"),
            line_start=chunk.start_char,
            line_end=chunk.payload.get("end_char", chunk.start_char + len(chunk.text)),
            content=chunk.text,
            source="vector",
            score=chunk.score,
        )
    )

initial_context = AccumulatedContext(items=initial_items)

agentic_result_1 = asyncio.run(
    test_agentic_loop(
        original_query=QUERY,
        checklist=query_expansion.checklist,
        context=initial_context,
        iteration=1,
    )
)

print(f"=== Agentic Loop Iteration 1 ===")
print(f"  All answered: {agentic_result_1.all_answered}")
print(f"  Updated checklist:")
for item in agentic_result_1.checklist:
    status = "✓" if item.answered else "○"
    print(f"    {status} [{item.id}] {item.question}")
print(f"  Planned retrieval queries ({len(agentic_result_1.retrieval_queries)}):")
for rq in agentic_result_1.retrieval_queries:
    print(f"    - [{rq.target.value}] {rq.query}")


# %% Cell 12 - STEP 4: Execute Retrieval Queries from Agentic Plan
async def execute_retrieval_queries(
    retrieval_queries: list, current_context: AccumulatedContext
):
    """Execute the retrieval queries planned by the agentic loop."""
    from backend.src.domain.schemas.retrieval import RetrievalTarget

    new_items = []

    for rq in retrieval_queries:
        if rq.target in (RetrievalTarget.VECTOR, RetrievalTarget.BOTH):
            filter_dict = None
            if rq.filters and rq.filters.file_ids:
                filter_dict = {"doc_ids": rq.filters.file_ids}

            chunks = await VectorService.search_collection(
                client=qdrant_client,
                collection_name="chunks",
                embed_config=embedder_config,
                query=rq.query,
                filters=filter_dict,
                limit=10,
            )

            for chunk in chunks:
                from backend.src.domain.schemas.retrieval import AccumulatedItem

                new_items.append(
                    AccumulatedItem(
                        file_id=chunk.payload.get("doc_id", ""),
                        path=chunk.payload.get("doc_title", "unknown"),
                        line_start=chunk.start_char,
                        line_end=chunk.payload.get(
                            "end_char", chunk.start_char + len(chunk.text)
                        ),
                        content=chunk.text,
                        source="vector",
                        score=chunk.score,
                    )
                )

        if (
            rq.target in (RetrievalTarget.GRAPH, RetrievalTarget.BOTH)
            and falkordb_client
        ):
            embedder = EmbedderService()
            embedded = await embedder.embed_query(rq.query, embedder_config)
            if embedded and embedded.dense:
                node_types = None
                if rq.filters and rq.filters.node_type:
                    type_map = {
                        "function": GraphNodeType.FUNCTION,
                        "class": GraphNodeType.CLASS,
                        "method": GraphNodeType.METHOD,
                    }
                    mapped = type_map.get(rq.filters.node_type.lower())
                    if mapped:
                        node_types = [mapped]

                file_ids = None
                if rq.filters and rq.filters.file_ids:
                    file_ids = [
                        int(fid) for fid in rq.filters.file_ids if fid.isdigit()
                    ]

                try:
                    result = await graph_service.semantic_search(
                        query_embedding=embedded.dense,
                        node_types=node_types,
                        file_ids=file_ids,
                        limit=10,
                    )

                    for node in result.nodes:
                        from backend.src.domain.schemas.retrieval import AccumulatedItem

                        new_items.append(node.to_accumulated_item())
                except Exception as e:
                    print(f"  Graph retrieval error: {e}")

    updated_context = current_context.add_items(new_items)
    return updated_context, new_items


if agentic_result_1.retrieval_queries:
    updated_context, new_items = asyncio.run(
        execute_retrieval_queries(
            agentic_result_1.retrieval_queries,
            initial_context,
        )
    )
    print(f"=== Retrieved {len(new_items)} new items ===")
    print(f"  Total context items: {len(updated_context.items)}")
else:
    updated_context = initial_context
    print("No retrieval queries to execute")

# %% Cell 13 - STEP 5: Agentic Loop Iteration 2
agentic_result_2 = asyncio.run(
    test_agentic_loop(
        original_query=QUERY,
        checklist=agentic_result_1.checklist,
        context=updated_context,
        iteration=2,
    )
)

print(f"=== Agentic Loop Iteration 2 ===")
print(f"  All answered: {agentic_result_2.all_answered}")
print(f"  Updated checklist:")
for item in agentic_result_2.checklist:
    status = "✓" if item.answered else "○"
    print(f"    {status} [{item.id}] {item.question}")
print(f"  Planned retrieval queries ({len(agentic_result_2.retrieval_queries)}):")
for rq in agentic_result_2.retrieval_queries:
    print(f"    - [{rq.target.value}] {rq.query}")


# %% Cell 14 - STEP 6: Final Rerank
def test_rerank_final(user_query: str, context: AccumulatedContext):
    """Rerank accumulated context against original query."""
    if not context.items:
        return context

    chunks = [
        RetrievedChunk(
            text=item.content,
            token_count=len(item.content.split()),
            start_char=item.line_start,
            score=item.score,
            payload={
                "doc_id": item.file_id,
                "doc_title": item.path,
            },
        )
        for item in context.items
    ]

    reranker = RerankerService(config=reranker_config)
    reranked = reranker.rerank(
        query=user_query,
        chunks=chunks,
        top_k=10,
    )

    reranked_items = []
    for chunk in reranked:
        for item in context.items:
            if item.content == chunk.text:
                reranked_items.append(item)
                break

    return AccumulatedContext(items=reranked_items)


final_context = test_rerank_final(QUERY, updated_context)
print(f"=== Final Reranked Context ===")
print(f"  Items after rerank: {len(final_context.items)}")
formatted = final_context.to_formatted_context()
print(f"  Formatted context preview:\n{formatted[:500]}...")


# %% Cell 15 - FULL PIPELINE: End-to-End Retrieval Pipeline
async def run_full_retrieval_pipeline(user_query: str, max_iterations: int = 3):
    """Run the complete retrieval pipeline end-to-end."""
    pipeline = build_retrieval_pipeline(
        llm_config=llm_config,
        embedder_config=embedder_config,
        vector_config=vector_store_config,
        reranker_config=reranker_config,
        falkordb_client=falkordb_client,
        token_budget=4096,
        max_iterations=max_iterations,
    )

    result = await pipeline.ainvoke({"user_query": user_query})
    return result


PIPELINE_QUERY = "How does filtering work in Qdrant collections?"
print(f"=== Running Full Retrieval Pipeline ===")
print(f"  Query: {PIPELINE_QUERY}")

start_time = time.time()
pipeline_result = asyncio.run(run_full_retrieval_pipeline(PIPELINE_QUERY))
elapsed = time.time() - start_time

print(f"\n  Pipeline completed in {elapsed:.2f}s")
print(f"  Iterations: {pipeline_result.get('iteration', 'N/A')}")
print(f"  File IDs: {pipeline_result.get('file_ids', set())}")
print(
    f"  Context items: {len(pipeline_result.get('accumulated_context', AccumulatedContext()).items)}"
)
print(
    f"\n  Formatted context preview:\n{pipeline_result.get('formatted_context', '')[:600]}..."
)

# %% Cell 16 - RETRIEVAL TOOL: Create and Test Tool
result_store = RetrievalResultStore()

retrieval_tool = create_retrieval_tool(
    llm_config=llm_config,
    embedder_config=embedder_config,
    vector_config=vector_store_config,
    reranker_config=reranker_config,
    falkordb_client=falkordb_client,
    token_budget=4096,
    max_iterations=3,
    result_store=result_store,
)

print(f"=== Retrieval Tool Created ===")
print(f"  Tool name: {retrieval_tool.name}")
print(f"  Tool description: {retrieval_tool.description[:100]}...")

# Test the tool directly
TOOL_TEST_QUERY = "What is vector search in Qdrant?"
print(f"\n=== Testing Retrieval Tool ===")
print(f"  Query: {TOOL_TEST_QUERY}")

start_time = time.time()
tool_result = await retrieval_tool.ainvoke({"query": TOOL_TEST_QUERY, "target": "both"})
elapsed = time.time() - start_time

print(f"  Completed in {elapsed:.2f}s")
print(f"  Result preview:\n{tool_result[:500]}...")
print(f"  Result store items: {len(result_store.items)}")

# %% Cell 17 - AGENT: Create FOSRA Agent
from backend.src.services.conversation.utils.prompts import FOSRA_AGENT_SYSTEM_PROMPT

agent, agent_result_store = create_fosra_agent(
    user_prefs=None,  # Uses defaults
    system_prompt=FOSRA_AGENT_SYSTEM_PROMPT,
)

print(f"=== FOSRA Agent Created ===")
print(f"  Agent type: {type(agent)}")
print(f"  Result store: {agent_result_store}")


# %% Cell 18 - UTILITY: Collection Stats
async def count_collections():
    parents = await VectorService.count_points(qdrant_async_client, "parents")
    chunks = await VectorService.count_points(qdrant_async_client, "chunks")
    return {"parents": parents, "chunks": chunks}


counts = asyncio.run(count_collections())
print("=== Collection Counts ===")
print(f"  Parents: {counts['parents']}")
print(f"  Chunks: {counts['chunks']}")

print("\n=== REPL Ready for Interactive Testing ===")
print("Available variables:")
print(f"  - qdrant_docs_result: Qdrant folder ingestion result")
print(f"  - single_file_result: Single file ingestion result")
print(f"  - codebase_result: Codebase ingestion result")
print(f"  - ingestion_stats: Summary of all ingested data")
print(f"  - query_expansion: Last query expansion result")
print(f"  - initial_retrieval_result: Initial dual_retrieve result")
print(f"  - agentic_result_1: First agentic loop iteration")
print(f"  - agentic_result_2: Second agentic loop iteration")
print(f"  - updated_context: Context after retrieval queries")
print(f"  - final_context: Reranked final context")
print(f"  - pipeline_result: Full pipeline result")
print(f"  - result_store: Retrieval result store")
print(f"  - retrieval_tool: The retrieval tool")
print(f"  - agent: The FOSRA agent")
print(f"  - agent_result_store: Agent's result store")
