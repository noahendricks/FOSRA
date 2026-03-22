# FOSRA Evolution Plan — Full Detail

## Current State Recap

**What exists and works:**
- FastAPI backend with SSE streaming chat
- PostgreSQL via SQLAlchemy (users, workspaces, convos, messages, docs, doc_topics)
- Qdrant vector store with hybrid retrieval (dense + sparse + late interaction)
- HiChunk hierarchical chunking (L1/L2/L3 with parent pointers, auto-merge)
- `code_chunker` library for code extraction (functions, classes, methods, imports — regex-based, multi-language)
- `ChunkerService` dispatching code vs text chunking
- `EmbedderService` via FastEmbed (dense/sparse/late)
- `RerankerService` via FlashRank
- LangGraph retrieval pipeline (reform → split → retrieve → coverage → retry)
- DeepAgents-based agent with `search_knowledge_base` tool
- Taskiq InMemoryBroker with processing tasks
- pydantic-settings for config (env vars)

**What's missing for FULL.md vision:**
- FalkorDB graph store (entire subsystem)
- Call graph extraction (tree-sitter for function call analysis within bodies)
- Cross-file import resolution + class hierarchy extraction
- TOML config system
- ~Multi-provider embedding (Ollama, API providers alongside FastEmbed)~ **SKIP THIS**
- ~Parent chunk summarization via LLM~ **SKIP THIS — parent text embedding is sufficient**
- Qdrant dual-collection structure (parents + chunks for coarse-to-fine routing)
- Unified RRF search pipeline
- Checklist-based query expansion
- Graph retrieval (semantic vector search + Cypher traversals on FalkorDB)
- Subagent with structured output for targeted retrieval queries
- Model registry (singleton, initialized once)
- Ingestion API endpoints
- Enhanced SSE events (phases, checklist updates, citations)
- `docs` table evolution (path, language, repo, source_type, checksum columns)

---

## Phase 1: Foundation — Config, Schema, Infrastructure

### 1.1 — TOML Config System

**New file: `config.toml`** (project root — default template)
```toml
[models]
embedding_local = "nomic-embed-text"
embedding_code_local = "nomic-embed-code"
embedding_api = "voyage-code-3"
llm_local = "llama3.1:8b"
llm_api = "claude-sonnet-4-20250514"
reranker = "ms-marco-MiniLM-L-6-v2"

[models.ops]
query_expansion = "local"
subagent = "local"
generation = "api"
classifier = "local"
summarization = "local"
code_embedding = "local"

[databases]
postgres = "postgresql+asyncpg://localhost:5432/fosra"
qdrant_url = "http://localhost:6333"

[databases.falkordb]
host = "localhost"
port = 6379

[ingestion]
chunk_size_parent = 768
chunk_size_child = 192
code_embedding_threshold = 0.4

[retrieval]
initial_summary_top_k = 20
initial_direct_top_k = 10
rerank_top_n = 15
max_iterations = 5
checklist_size = 5

[api]
host = "0.0.0.0"
port = 8000
```

**New file: `backend/src/settings/config_loader.py`**
- `load_config(path: str | None = None) -> dict` — loads TOML from:
  1. Explicit path argument
  2. `FOSRA_CONFIG` env var
  3. `./config.toml` (project root)
  4. `~/.config/fosra/config.toml`
- Uses `tomllib` (stdlib 3.11+) for parsing
- Returns raw dict, consumed by pydantic settings as override source
- Secrets (API keys) stay in env vars only — never in TOML

**Edit: `backend/src/settings/__init__.py`**
- Add new settings groups:
  ```python
  class FalkorDBSettings(BaseSettings):
      host: str = "localhost"
      port: int = 6379

  class ModelOpsSettings(BaseSettings):
      query_expansion: str = "local"  # local | api
      subagent: str = "local"
      generation: str = "api"
      classifier: str = "local"
      summarization: str = "local"
      code_embedding: str = "local"

  class IngestionSettings(BaseSettings):
      chunk_size_parent: int = 768
      chunk_size_child: int = 192
      code_embedding_threshold: float = 0.4

  class RetrievalSettings(BaseSettings):
      initial_summary_top_k: int = 20
      initial_direct_top_k: int = 10
      rerank_top_n: int = 15
      max_iterations: int = 5
      checklist_size: int = 5
  ```
- Add `falkordb`, `model_ops`, `ingestion`, `retrieval` fields to main `Settings`
- Integrate with `config_loader.py`: on `Settings` init, load TOML first, then override with env vars

**Edit: `backend/src/domain/schemas/config.py`**
- Add `FalkorDBConfig`, `ModelOpsConfig` domain schemas (Pydantic Models)
- These mirror the settings classes but are domain-layer types passed to services

**Edit: `backend/src/domain/enums.py`**
- Add `EmbeddingProvider` enum: `FASTEMBED`, `OLLAMA`, `API`
- Add `SourceType` enum: `CODEBASE`, `DOC` (for `docs.source_type` column)
- Add `GraphNodeType` enum: `FILE`, `MODULE`, `FUNCTION`, `CLASS`, `METHOD`

---

### 1.2 — Evolve `docs` Table (PostgreSQL)

**New Alembic migration**
```sql
ALTER TABLE docs ADD COLUMN path TEXT;
ALTER TABLE docs ADD COLUMN language TEXT;
ALTER TABLE docs ADD COLUMN repo TEXT;
ALTER TABLE docs ADD COLUMN source_type TEXT NOT NULL DEFAULT 'doc';
ALTER TABLE docs ADD COLUMN checksum TEXT;
ALTER TABLE docs ADD CONSTRAINT uq_docs_path_repo UNIQUE (path, repo);
```

**Edit: `backend/src/storage/models.py`** — add columns to `DocORM`:
```python
path = Column(Text, nullable=True)
language = Column(Text, nullable=True)
repo = Column(Text, nullable=True)
source_type = Column(Text, nullable=False, default="doc")
checksum = Column(Text, nullable=True)
```

**Edit: `backend/src/domain/schemas/doc.py`** — add matching fields to `Doc` domain struct:
```python
path: str | None = None
language: str | None = None
repo: str | None = None
source_type: str = "doc"
checksum: str | None = None
```

---

### 1.3 — FalkorDB Infrastructure

**Add `falkordb` to `pyproject.toml`**

**Edit: `backend/src/api/lifecycle.py`** — add FalkorDB to `Infrastructure`:
```python
class Infrastructure:
    qdrant_client: AsyncQdrantClient
    session_factory: async_sessionmaker
    falkordb_graph: Graph  # FalkorDB graph handle

async def init_infrastructure():
    # ... existing qdrant + postgres init ...
    db = FalkorDB(host=settings.falkordb.host, port=settings.falkordb.port)
    graph = db.select_graph("fosra")
    global_infra.falkordb_graph = graph
```

**Graph index initialization** (called once on first startup or via explicit init endpoint):
```python
def ensure_graph_indexes(graph):
    # range indexes for fast lookups
    graph.create_node_range_index("File", "doc_id", "path", "language")
    graph.create_node_range_index("Function", "doc_id", "name", "line_start")
    graph.create_node_range_index("Class", "doc_id", "name", "line_start")
    graph.create_node_range_index("Module", "doc_id", "name")

    # fulltext for name search
    graph.create_node_fulltext_index("Function", "name")
    graph.create_node_fulltext_index("Class", "name")

    # vector indexes (dimension depends on embedding model)
    graph.create_node_vector_index("Function", "embedding",
        dim=768, similarity_function="cosine")
    graph.create_node_vector_index("Class", "embedding",
        dim=768, similarity_function="cosine")
```

---

### [SKIP] ~1.4 — Embedding Provider Abstraction [SKIP THIS -- FASTEMBED KEPT AS ONLY AND PRIMARY EMBEDDER]~

**Edit: `backend/src/services/processing/embedder_service.py`**

***!SKIP***


***!SKIP***

---

## Phase 2: Codebase Ingestion Pipeline (FalkorDB)

### 2.1 — Supplemental Tree-sitter Analysis (Call Graph + Signatures + Hierarchy)

`code_chunker` extracts function/class/method/import nodes but does NOT extract:
- Function calls within bodies (needed for `:CALLS` edges)
- Function signatures (params, return types)
- Class hierarchy (base classes, implements)
- Cross-file import resolution

Tree-sitter fills these specific gaps.

**New file: `backend/src/services/processing/callgraph_service.py`**

```python
class CallGraphService:
    """Extracts call relationships and supplemental metadata
    that code_chunker doesn't provide."""

    parsers: dict[str, tree_sitter.Parser]  # one per language

    def extract_calls(self, code: str, language: str) -> list[CallEdge]:
        """Walk function bodies, find call expressions,
        return (caller_name, callee_name) pairs."""
        # tree-sitter query per language:
        # Python:  (call function: (identifier) @callee)
        # Go:      (call_expression function: (identifier) @callee)
        # JS/TS:   (call_expression function: (identifier) @callee)
        # Rust:    (call_expression function: (identifier) @callee)
        ...

    def extract_signatures(self, code: str, language: str) -> dict[str, Signature]:
        """Extract function parameter names/types and return types."""
        # tree-sitter queries for parameter lists and return type annotations
        ...

    def extract_class_hierarchy(self, code: str, language: str) -> list[InheritanceEdge]:
        """Extract base classes / implements / extends relationships."""
        # Python:  (class_definition superclasses: (argument_list (identifier) @base))
        # Go:      (struct_type (field_declaration_list (field_declaration type: (type_identifier) @embed)))
        # JS/TS:   (class_declaration (class_heritage (extends_clause (identifier) @base)))
        # Rust:    impl blocks with trait
        ...

    def resolve_imports(
        self, imports: list[Import], file_path: str, repo_root: str
    ) -> list[ResolvedImport]:
        """Resolve import module strings to actual file paths within the repo."""
        # Python:  'from foo.bar import baz' → find foo/bar.py or foo/bar/__init__.py
        # Go:      'import "github.com/x/y/pkg"' → find pkg/ directory
        # JS/TS:   'import { x } from "./utils"' → find utils.ts/utils.js/utils/index.ts
        # Rust:    'use crate::foo::bar' → find src/foo/bar.rs or src/foo/bar/mod.rs
        ...
```

**New file: `backend/src/domain/schemas/graph.py`**
```python
class CallEdge(BaseModel):
    caller_name: str
    callee_name: str
    caller_file: str | None = None
    callee_file: str | None = None

class InheritanceEdge(BaseModel):
    child_name: str
    parent_name: str

class ResolvedImport(BaseModel):
    source_file: str
    target_file: str
    imported_names: list[str]

class Signature(BaseModel):
    name: str
    params: list[tuple[str, str | None]]  # (name, type_annotation)
    return_type: str | None = None
```

**Add to `pyproject.toml`:**
- `tree-sitter`
- `tree-sitter-python`
- `tree-sitter-javascript`
- `tree-sitter-typescript`
- `tree-sitter-go`
- `tree-sitter-rust`

---

### 2.2 — FalkorDB Graph Construction Service

**New file: `backend/src/services/processing/graph_service.py`**

Consumes `code_chunker.ParseResult` + `CallGraphService` output → FalkorDB graph.

```python
class GraphService:
    graph: Graph  # FalkorDB graph handle from Infrastructure

    def ensure_indexes(self):
        """Create indexes if they don't exist. Idempotent."""
        ...

    async def upsert_file(self, parse_result: ParseResult, doc_id: int, repo: str):
        """Create :File node + :Function, :Class, :Method nodes + structural edges."""
        # 1. CREATE (:File { doc_id, path, language, repo })
        # 2. For each chunk in parse_result.chunks:
        #    - FUNCTION → CREATE (:Function { doc_id, name, line_start, line_end, code, signature, docstring })
        #                 CREATE (:File)-[:CONTAINS]->(:Function)
        #    - CLASS    → CREATE (:Class { doc_id, name, line_start, line_end, code })
        #                 CREATE (:File)-[:CONTAINS]->(:Class)
        #    - METHOD   → CREATE (:Function { doc_id, name, line_start, line_end, code, class_name })
        #                 MATCH (c:Class { name: class_name, doc_id })
        #                 CREATE (c)-[:HAS_METHOD]->(method)
        # 3. Merge signatures from CallGraphService onto Function nodes
        ...

    async def upsert_call_edges(self, call_edges: list[CallEdge], doc_id: int):
        """Create :CALLS edges between Function nodes."""
        # MATCH (caller:Function { name: $caller_name, doc_id: $doc_id })
        # MATCH (callee:Function { name: $callee_name })
        # MERGE (caller)-[:CALLS]->(callee)
        ...

    async def upsert_import_edges(self, imports: list[ResolvedImport]):
        """Create :IMPORTS edges between Module/File nodes."""
        # MATCH (src:File { path: $source_file })
        # MATCH (tgt:File { path: $target_file })
        # MERGE (src)-[:IMPORTS { names: $imported_names }]->(tgt)
        ...

    async def upsert_inheritance_edges(self, edges: list[InheritanceEdge], doc_id: int):
        """Create :INHERITS edges between Class nodes."""
        # MATCH (child:Class { name: $child_name, doc_id: $doc_id })
        # MATCH (parent:Class { name: $parent_name })
        # MERGE (child)-[:INHERITS]->(parent)
        ...

    async def embed_code_nodes(self, doc_id: int, embedder: EmbedderService):
        """Embed Function and Class nodes, store embedding on node."""
        # 1. MATCH (n) WHERE n.doc_id = $doc_id AND (n:Function OR n:Class)
        #    RETURN n.name, n.code, n.signature, n.docstring, ID(n)
        # 2. For each node: embed(signature + docstring + first N lines of body)
        # 3. SET n.embedding = $embedding_vector
        ...

    async def delete_file_graph(self, doc_id: int):
        """Remove all nodes/edges for a file (for re-index)."""
        # MATCH (n { doc_id: $doc_id }) DETACH DELETE n
        ...
```

**Node properties (final schema):**
```cypher
(:File     { doc_id, path, language, repo })
(:Module   { doc_id, name, path })
(:Class    { doc_id, name, line_start, line_end, code, embedding })
(:Function { doc_id, name, line_start, line_end, code, signature, docstring, class_name, embedding })
```

**Edge types:**
```cypher
(:File)-[:CONTAINS]->(:Function)
(:File)-[:CONTAINS]->(:Class)
(:Class)-[:HAS_METHOD]->(:Function)
(:Function)-[:CALLS]->(:Function)
(:File)-[:IMPORTS]->(:File)
(:Class)-[:INHERITS]->(:Class)
```

---

### 2.3 — Codebase Ingestion Orchestrator

**New file: `backend/src/tasks/codebase_ingestion.py`**

End-to-end pipeline for indexing a codebase into FalkorDB.

```python
async def ingest_codebase(path: str, repo: str, settings: Settings):
    """Walk directory → register files → parse → call graph → graph upsert → embed."""

    chunker = CodeChunker(config=CodeChunkerConfig(
        include_imports=True, include_comments=True
    ))
    callgraph = CallGraphService()
    graph_svc = GraphService(graph=get_infra().falkordb_graph)
    embedder = EmbedderService()

    # STEP 1: walk directory, collect source files
    source_files = walk_source_files(path, extensions=SUPPORTED_EXTENSIONS)

    # STEP 2: register/check files in docs table
    async with get_session() as session:
        for file_path in source_files:
            checksum = sha256_file(file_path)
            existing = await get_doc_by_path(session, file_path, repo)
            if existing and existing.checksum == checksum:
                continue  # unchanged, skip
            doc = await upsert_doc(session, file_path, repo, checksum, source_type="codebase")
            files_to_process.append((file_path, doc.doc_id))

    # STEP 3: parse each file with code_chunker
    all_parse_results: dict[int, ParseResult] = {}
    for file_path, doc_id in files_to_process:
        result = chunker.parse_file(file_path)
        all_parse_results[doc_id] = result

    # STEP 4: extract call graph + signatures + hierarchy via tree-sitter
    all_call_edges: list[CallEdge] = []
    all_inheritance: list[InheritanceEdge] = []
    for doc_id, parse_result in all_parse_results.items():
        code = parse_result.raw_code
        lang = parse_result.language

        calls = callgraph.extract_calls(code, lang)
        all_call_edges.extend(calls)

        hierarchy = callgraph.extract_class_hierarchy(code, lang)
        all_inheritance.extend(hierarchy)

        sigs = callgraph.extract_signatures(code, lang)
        # attach signatures to parse_result chunks (enrich metadata)

    # STEP 5: resolve cross-file imports
    all_imports = []
    for doc_id, pr in all_parse_results.items():
        resolved = callgraph.resolve_imports(pr.imports, pr.file_path, path)
        all_imports.extend(resolved)

    # STEP 6: upsert to FalkorDB
    for doc_id, pr in all_parse_results.items():
        await graph_svc.upsert_file(pr, doc_id, repo)

    await graph_svc.upsert_call_edges(all_call_edges)
    await graph_svc.upsert_import_edges(all_imports)
    await graph_svc.upsert_inheritance_edges(all_inheritance)

    # STEP 7: embed code nodes
    for doc_id in all_parse_results:
        await graph_svc.embed_code_nodes(doc_id, embedder)
```

**Edit: `backend/src/tasks/processing.py`** — register `ingest_codebase` as Taskiq task

---

## Phase 3: Document Ingestion Evolution (Qdrant)

### 3.1 — Qdrant Dual-Collection Structure

**Edit: `backend/src/services/retrieval/vector_service.py`**

Current state: single collection with all vectors. Target: two collections.

**Key insight:** We use parent chunk text directly as the "coarse" vector — no LLM summarization. For technical docs and code, embeddings on the original parent text capture ~90% of summary value at $0 cost. The hierarchical pattern (search coarse, return fine) works regardless of whether "coarse" is a summary or the original parent text.

```python
PARENTS_COLLECTION = "parents"   # L1/L2 chunks (coarse, for routing)
CHUNKS_COLLECTION = "chunks"     # L3 leaf chunks (fine, for retrieval)

class VectorService:
    async def ensure_collections(self):
        """Create both collections with dense + sparse + late interaction configs."""
        for name in [PARENTS_COLLECTION, CHUNKS_COLLECTION]:
            await self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE),
                    "late": VectorParams(size=128, distance=Distance.COSINE, multivector_config=...),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )

    async def upsert_parents(self, parents: list[Chunk]):
        """Upsert parent chunk vectors (L1/L2) to parents collection.
        
        Uses parent.text directly — no LLM summarization.
        Payload: { doc_id, chunk_id, line_start, line_end, level, parent_text }
        """
        ...

    async def upsert_chunks(self, chunks: list[Chunk]):
        """Upsert leaf chunk vectors (L3) to chunks collection.
        
        Payload: { doc_id, chunk_id, parent_id, line_start, line_end, parent_text (for auto-merge) }
        """
        ...

    async def rrf_search(self, query_vectors, collection: str, filters, top_k: int) -> list:
        """RRF fusion over dense + sparse + late interaction vectors."""
        dense_results = await self.client.query_points(collection, query=query_vectors.dense, ...)
        sparse_results = await self.client.query_points(collection, query=query_vectors.sparse, ...)
        late_results = await self.client.query_points(collection, query=query_vectors.late, ...)
        return rrf_merge([dense_results, sparse_results, late_results], k=60)
```

- `rrf_merge()` — reciprocal rank fusion with constant k=60
- Existing auto-merge retrieval logic preserved for chunks collection
- Search methods updated to accept collection name parameter
- Parent collection enables coarse-to-fine routing without LLM cost

---

### 3.2 — Document Ingestion Pipeline Update

**Edit: `backend/src/tasks/storing.py`** — update `store_file_vectors`:

```python
async def store_file_vectors(docs: list[Doc], config):
    """Full doc ingestion: register → chunk → embed → upsert to dual collections."""

    # step 1: register files in docs table
    for doc in docs:
        checksum = sha256(doc.page_content)
        await upsert_doc(session, doc, source_type="doc", checksum=checksum)

    # step 2: chunk via existing HiChunk (L1/L2/L3)
    chunks_per_doc = await ChunkerService.chunk_documents(docs, config.chunker)

    for doc_chunks in chunks_per_doc:
        # step 3: separate by level
        parent_chunks = [c for c in doc_chunks if c.metadata.level in (1, 2)]  # L1/L2
        leaf_chunks = [c for c in doc_chunks if c.metadata.level == 3]        # L3

        # step 4: embed parent chunks → upsert to parents collection
        if parent_chunks:
            await embedder.embed_chunks(parent_chunks, config.embedder)
            await vector_service.upsert_parents(parent_chunks)

        # step 5: embed leaf chunks → upsert to chunks collection
        if leaf_chunks:
            await embedder.embed_chunks(leaf_chunks, config.embedder)
            await vector_service.upsert_chunks(leaf_chunks)
```

**No new files needed** — existing `EmbedderService` and `VectorService` handle the dual-collection logic.

---

## Phase 4: Retrieval Pipeline Evolution

### 4.1 — Graph Retriever

**New file: `backend/src/services/retrieval/graph_retriever.py`**

```python
class GraphRetriever:
    """Retrieves code context from FalkorDB graph."""

    graph: Graph  # from Infrastructure

    async def semantic_search(
        self, query_embedding: list[float], filters: dict, top_k: int = 20
    ) -> list[GraphResult]:
        """Vector similarity search on embedded Function/Class nodes."""
        # CALL db.idx.vector.queryNodes('Function', 'embedding', $top_k, vecf32($emb))
        # YIELD node, score
        # WHERE node.doc_id IN $doc_ids  (if filtered)
        # RETURN node, score
        ...

    async def get_callers(self, function_name: str, depth: int = 1) -> list[GraphResult]:
        """Who calls this function?"""
        # MATCH (caller:Function)-[:CALLS*1..{depth}]->(f:Function {name: $name})
        # RETURN caller, f
        ...

    async def get_call_chain(self, function_name: str, depth: int = 5) -> list[GraphResult]:
        """Full outbound call chain from a function."""
        # MATCH path = (f:Function {name: $name})-[:CALLS*1..{depth}]->(dep)
        # RETURN path
        ...

    async def get_file_symbols(self, doc_id: int) -> list[GraphResult]:
        """All symbols defined in a file."""
        # MATCH (file:File {doc_id: $doc_id})-[:CONTAINS]->(node)
        # RETURN node
        ...

    async def get_inheritance_chain(self, class_name: str, depth: int = 10) -> list[GraphResult]:
        """Inheritance chain for a class."""
        # MATCH path = (c:Class {name: $name})-[:INHERITS*1..{depth}]->(base)
        # RETURN path
        ...

    async def get_file_imports(self, doc_id: int) -> list[GraphResult]:
        """What files does this file import?"""
        # MATCH (f:File {doc_id: $doc_id})-[:IMPORTS]->(imported:File)
        # RETURN imported
        ...
```

**New schema: `GraphResult`** in `backend/src/domain/schemas/graph.py`:
```python
class GraphResult(BaseModel):
    node_type: str  # Function, Class, File
    name: str
    doc_id: int
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    code: str | None = None
    score: float | None = None  # for semantic results
```

---

### 4.2 — Query Expansion with Checklist

**Edit: `backend/src/services/conversation/query_service.py`**

Extend `reform_query()` to produce structured checklist output:

```python
async def expand_query(self, user_query: str) -> QueryExpansion:
    """Generate rewritten query + checklist via structured output."""
    # uses tool calling / JSON schema for strict output:
    # {
    #   "rewritten_query": "...",
    #   "checklist": [
    #     { "id": 1, "question": "...", "answered": false },
    #     ...
    #   ]
    # }
    ...
```

**New file: `backend/src/domain/schemas/query.py`**
```python
class ChecklistItem(BaseModel):
    id: int
    question: str
    answered: bool = False

class QueryExpansion(BaseModel):
    rewritten_query: str
    checklist: list[ChecklistItem]

class RetrievalQuery(BaseModel):
    query: str
    target: str  # "vector" | "graph" | "both"
    filters: dict | None = None

class SubagentInput(BaseModel):
    user_query: str
    checklist: list[ChecklistItem]
    accumulated_context: list[dict]
    iteration: int

class SubagentOutput(BaseModel):
    checklist: list[ChecklistItem]
    all_answered: bool
    retrieval_queries: list[RetrievalQuery]
```

---

### 4.3 — Evolve LangGraph Retrieval Pipeline

**Edit: `backend/src/services/conversation/retrieval_pipeline.py`**

Current pipeline: `reform_query → split_subqueries → retrieve → check_coverage → retry`

Target pipeline:
```
expand_query → parallel_initial_retrieval → subagent_loop → final_rerank
                    ├── summary_search (Qdrant summaries)
                    └── direct_search (Qdrant chunks)
```

**State changes:**
```python
class PipelineState(TypedDict):
    user_query: str
    expansion: QueryExpansion           # NEW: replaces reformulated_query
    checklist: list[ChecklistItem]      # NEW: replaces subqueries
    accumulated_context: list[dict]     # NEW: structured context objects
    iteration: int                      # NEW
    file_ids: list[int]                 # NEW: scoped file IDs
    final_results: list[dict]           # reranked results
```

**New nodes:**
1. `expand_query` — calls `QueryService.expand_query()`, produces `expansion` + `checklist`
2. `initial_retrieval` — runs in parallel:
   - `VectorService.rrf_search()` on summaries collection → file_id set
   - `VectorService.rrf_search()` on chunks collection → file_id set
   - Merge + deduplicate file_ids
3. `subagent` — single-turn LLM call that:
   - Receives: user_query, checklist, accumulated_context, iteration
   - Outputs: updated checklist (items marked answered), retrieval_queries
   - Each `retrieval_query` specifies `target: vector | graph | both`
4. `execute_retrieval` — dispatches retrieval queries:
   - `target=vector` → `VectorService.rrf_search()` on chunks collection with filters
   - `target=graph` → `GraphRetriever.semantic_search()` or `.get_callers()` etc.
   - `target=both` → both in parallel
5. `accumulate_context` — deduplicates new results into accumulated_context
6. `check_complete` — if `all_answered` or `iteration >= max_iterations`, route to `final_rerank`; else loop back to `subagent`
7. `final_rerank` — `RerankerService.rerank()` on accumulated_context against original query

**Conditional edge:** `check_complete` → `subagent` (loop) or `final_rerank` (done)

---

### 4.4 — Update Retrieval Tools

**Edit: `backend/src/services/conversation/tools.py`**

Update `search_knowledge_base` tool to support graph queries:

```python
@tool
async def search_knowledge_base(query: str, target: str = "vector", filters: dict = None):
    """Search the knowledge base.

    Args:
        query: search query
        target: "vector" for document search, "graph" for code structure,
                "both" for combined search
        filters: optional filters (file_ids, node_type, language)
    """
    results = []
    if target in ("vector", "both"):
        results.extend(await vector_service.rrf_search(...))
    if target in ("graph", "both"):
        results.extend(await graph_retriever.semantic_search(...))
    return reranker.rerank(results, query)
```

**Edit: `backend/src/services/conversation/utils/prompts.py`**
- Update subagent prompt to emit structured `SubagentOutput` JSON
- Add graph-aware instructions (when to target graph vs vector)
- Add checklist assessment instructions

---

## Phase 5: Model Registry + Streaming Enhancements

### 5.1 — Model Registry

**New file: `backend/src/services/model_registry.py`**

```python
class ModelRegistry:
    """Singleton holding initialized model instances. Loaded once at startup."""

    _instance: ModelRegistry | None = None

    # embedding providers
    fastembed_provider: FastEmbedProvider | None
    ollama_provider: OllamaProvider | None
    api_provider: APIProvider | None

    # reranker
    reranker: FlashRankReranker | None

    # sparse/late encoders (always FastEmbed)
    sparse_encoder: SparseTextEmbedding | None
    late_encoder: LateInteractionTextEmbedding | None

    @classmethod
    async def initialize(cls, settings: Settings) -> ModelRegistry:
        """Initialize all configured models. Called once at startup."""
        registry = cls()
        registry.fastembed_provider = FastEmbedProvider(settings.embedding)

        if settings.model_ops.uses_ollama:
            registry.ollama_provider = OllamaProvider(settings)

        if settings.api_keys.has_embedding_api:
            registry.api_provider = APIProvider(settings)

        registry.reranker = FlashRankReranker(settings.reranker)
        registry.sparse_encoder = SparseTextEmbedding(model_name=settings.embedding.sparse_model)
        registry.late_encoder = LateInteractionTextEmbedding(model_name=settings.embedding.late_model)

        cls._instance = registry
        return registry

    @classmethod
    def get(cls) -> ModelRegistry:
        if cls._instance is None:
            raise RuntimeError("ModelRegistry not initialized")
        return cls._instance

    def get_embedder(self, operation: str) -> EmbeddingProvider:
        """Dispatch to correct provider based on [models.ops] config."""
        provider_type = self.settings.model_ops.get_provider(operation)
        match provider_type:
            case "local": return self.fastembed_provider
            case "ollama": return self.ollama_provider
            case "api": return self.api_provider
```

**Edit: `backend/src/api/lifecycle.py`** — initialize registry at startup:
```python
async def startup():
    await init_infrastructure()
    await ModelRegistry.initialize(settings)
```

---

### 5.2 — SSE Streaming Enhancements

**Edit: `backend/src/api/routes/workspace.py`** (`send_message_stream`)

Add structured SSE events for each pipeline phase:

```python
async def send_message_stream(request):
    # phase 1: expansion
    yield sse_event("status", {"phase": "expanding", "message": "Expanding query..."})
    expansion = await query_service.expand_query(request.query)
    yield sse_event("expansion", {
        "rewritten_query": expansion.rewritten_query,
        "checklist": [item.to_dict() for item in expansion.checklist]
    })

    # phase 2: initial retrieval
    yield sse_event("status", {"phase": "retrieving", "message": "Searching..."})
    file_ids = await initial_retrieve(expansion.rewritten_query)
    yield sse_event("file_ids", {"count": len(file_ids)})

    # phase 3: agentic loop
    async for event in agentic_loop(request.query, expansion, file_ids):
        yield sse_event(event.type, event.data)
        # event types: "checklist_update", "iteration", "status"

    # phase 4: generation
    yield sse_event("status", {"phase": "generating", "message": "Generating response..."})
    async for token in llm.stream(context, request.query):
        yield sse_event("token", {"content": token})

    # citations emitted inline during generation
    for citation in collected_citations:
        yield sse_event("citation", {
            "doc_id": citation.doc_id,
            "path": citation.path,
            "line_start": citation.line_start,
            "line_end": citation.line_end
        })

    yield sse_event("done", {})
```

SSE event types:
- `status` — phase label + human message
- `expansion` — rewritten query + checklist for rendering
- `file_ids` — initial scope count
- `checklist_update` — per iteration, which items answered
- `iteration` — current loop number
- `citation` — file reference with line range
- `token` — LLM generation token
- `done` — stream complete

---

## Phase 6: Ingestion API Endpoints

### 6.1 — Ingestion Routes

**Edit: `backend/src/api/routes/search.py`** (or new file `backend/src/api/routes/ingestion.py`)

```python
@router.post("/index/docs")
async def index_docs(request: IndexRequest):
    """Index documentation directory into Qdrant."""
    task = await ingest_docs.kiq(request.path)
    return {"task_id": str(task.task_id), "status": "started"}

@router.post("/index/code")
async def index_code(request: IndexRequest):
    """Index codebase directory into FalkorDB."""
    task = await ingest_codebase.kiq(request.path, request.repo or "default")
    return {"task_id": str(task.task_id), "status": "started"}

@router.post("/index/all")
async def index_all(request: IndexRequest):
    """Auto-detect file types, index docs to Qdrant, code to FalkorDB."""
    # classify files by extension, dispatch to appropriate pipeline
    ...

@router.post("/reindex/{index_type}")
async def reindex(index_type: str, request: IndexRequest):
    """Drop + rebuild index for docs or code."""
    if index_type == "docs":
        await vector_service.delete_collection(SUMMARIES_COLLECTION)
        await vector_service.delete_collection(CHUNKS_COLLECTION)
        await ingest_docs.kiq(request.path)
    elif index_type == "code":
        await graph_service.clear_graph(request.repo)
        await ingest_codebase.kiq(request.path, request.repo)

@router.get("/index/status")
async def index_status():
    """Return index stats: file count, node count, chunk count."""
    doc_count = await count_docs(session)
    node_count = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    chunk_count = await vector_service.count_points(CHUNKS_COLLECTION)
    summary_count = await vector_service.count_points(SUMMARIES_COLLECTION)
    return {
        "files": doc_count,
        "graph_nodes": node_count,
        "chunks": chunk_count,
        "summaries": summary_count,
    }
```

**New schemas in `backend/src/api/schemas/`:**
```python
class IndexRequest(BaseModel):
    path: str
    repo: str | None = None
```

---

## Dependency Additions (`pyproject.toml`)

| Package | Purpose | Required? |
|---------|---------|-----------|
| `falkordb` | FalkorDB Python client | Yes |
| `tree-sitter` | AST parsing core (call graph extraction) | Yes |
| `tree-sitter-python` | Python grammar | Yes |
| `tree-sitter-javascript` | JS grammar | Yes |
| `tree-sitter-typescript` | TS grammar | Yes |
| `tree-sitter-go` | Go grammar | Yes |
| `tree-sitter-rust` | Rust grammar | Yes |

---

## Files Summary

### New Files (10)
| File | Phase | Purpose |
|------|-------|---------|
| `config.toml` | 1.1 | User-facing TOML config template |
| `backend/src/settings/config_loader.py` | 1.1 | TOML → pydantic settings bridge |
| `backend/src/domain/schemas/graph.py` | 2.1 | CallEdge, InheritanceEdge, ResolvedImport, Signature, GraphResult |
| `backend/src/domain/schemas/query.py` | 4.2 | QueryExpansion, ChecklistItem, SubagentInput/Output, RetrievalQuery |
| `backend/src/services/processing/callgraph_service.py` | 2.1 | Tree-sitter call graph + signatures + hierarchy |
| `backend/src/services/retrieval/graph_service.py` | 2.2 | FalkorDB graph construction + code embedding |
| `backend/src/services/retrieval/graph_retriever.py` | 4.1 | FalkorDB semantic + structural retrieval |
| `backend/src/services/model_registry.py` | 5.1 | Singleton model/embedder registry |
| `backend/src/tasks/codebase_ingestion.py` | 2.3 | Codebase ingestion orchestrator |
| `backend/src/api/routes/ingestion.py` | 6.1 | Ingestion API endpoints (optional — could go in search.py) |

### Edited Files (16)
| File | Phase | Changes |
|------|-------|---------|
| `pyproject.toml` | 1.1 | Add falkordb, tree-sitter-*, httpx |
| `config.toml` | 1.1 | New file — default config template |
| `backend/src/settings/__init__.py` | 1.1 | FalkorDB, ModelOps, Ingestion, Retrieval settings |
| `backend/src/domain/schemas/config.py` | 1.1 | FalkorDBConfig, ModelOpsConfig domain schemas |
| `backend/src/domain/enums.py` | 1.1 | EmbeddingProvider, SourceType, GraphNodeType enums |
| `backend/src/storage/models.py` | 1.2 | DocORM: path, language, repo, source_type, checksum columns |
| `backend/src/domain/schemas/doc.py` | 1.2 | Doc: matching new fields |
| `backend/src/api/lifecycle.py` | 1.3 | FalkorDB client init + model registry init |
| ~`backend/src/services/processing/embedder_service.py`~ | SKIP ~| ~Multi-provider abstraction~  **|
| `backend/src/services/retrieval/vector_service.py` | 3.1 | Dual collections (parents + chunks), RRF fusion |
| `backend/src/tasks/storing.py` | 3.2 | Updated doc ingestion for dual collections |
| `backend/src/tasks/processing.py` | 2.3 | Register ingest_codebase task |
| `backend/src/services/conversation/query_service.py` | 4.2 | expand_query() with checklist |
| `backend/src/services/conversation/retrieval_pipeline.py` | 4.3 | Full pipeline rewrite with graph + checklist + subagent |
| `backend/src/services/conversation/tools.py` | 4.4 | Graph query support in search tool |
| `backend/src/services/conversation/utils/prompts.py` | 4.4 | Subagent + graph-aware prompts |
| `backend/src/api/routes/workspace.py` | 5.2 | SSE phase events, checklist updates, citations |

---

## Execution Order (Critical Path)
```
Phase 1.1 (TOML config) ─────────────────────────┐
Phase 1.2 (docs table migration) ────────────────┤
Phase 1.3 (FalkorDB infra) ──────────────────────┤
  ───────────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                    ▼
Phase 2.1 (callgraph svc)   Phase 3.1 (Qdrant dual collections)
Phase 2.2 (graph svc)
Phase 2.3 (codebase ingest)
          │                    │
          └─────────┬──────────┘
                    ▼
Phase 4.1 (graph retriever)
Phase 4.2 (query expansion + checklist)
Phase 4.3 (pipeline evolution)
Phase 4.4 (tools + prompts)
                    │
                    ▼
Phase 5.1 (model registry)
Phase 5.2 (SSE enhancements)
                    │
                    ▼
Phase 6.1 (ingestion endpoints)
```

Phases 2 and 3 are independent branches that can be built in parallel after Phase 1. Phase 4 requires both 2 and 3 to be complete. Phases 5 and 6 are polish and can be done last.
