# SOTA RAG TUI — Implementation Plan

## Overview

A developer-focused RAG (Retrieval-Augmented Generation) TUI application that combines graph-based codebase indexing with hierarchical vector-based document retrieval. The system exposes a Python/FastAPI backend to a Go/Bubbletea TUI client. Codebases are stored as property graphs in FalkorDB. Technical and code documentation is stored as hierarchical vector embeddings in Qdrant. PostgreSQL serves as the canonical source of truth for file metadata, tying both stores together via `file_id`.

---

## Tech Stack

### Backend
- **Language**: Python 3.12+
- **API**: FastAPI (async, SSE for streaming)
- **Task Queue**: Taskiq (asyncio workers, persistent model workers)
- **Graph DB**: FalkorDB (codebase graphs, code embeddings on nodes)
- **Vector DB**: Qdrant (docs + code-in-docs, RRF pipeline)
- **Relational DB**: PostgreSQL (canonical file metadata, file_id registry)
- **Reranker**: FlashRerank
- **Embeddings**: Hybrid — local (Ollama: `nomic-embed-code`, `nomic-embed-text`) + API (`voyage-code-3` for code nodes, `text-embedding-3-large` fallback)
- **LLM**: Configurable — local (Ollama) or API (OpenAI/Anthropic), user-selectable per operation
- **AST Parsing**: Tree-sitter (Python bindings: `tree-sitter`, `tree-sitter-languages`)
- **Call Graph**: `pyan3` (Python), language-specific static analyzers per language
- **Structured Output**: LLM tool calling with strict JSON schema for agent query emission

### Frontend (TUI)
- **Language**: Go 1.22+
- **Framework**: Bubbletea
- **Styling**: Lipgloss
- **Components**: Bubbles
- **API Client**: SSE stream reader via `http.Response.Body` → `tea.Cmd` → `tea.Msg`

---

```

---

## PostgreSQL Schema

### `files` table — canonical file registry
```sql
CREATE TABLE files (
    id          SERIAL PRIMARY KEY,
    path        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    language    TEXT,                    -- go, python, rust, markdown etc.
    repo        TEXT,
    source_type TEXT NOT NULL,           -- 'codebase' | 'doc'
    indexed_at  TIMESTAMPTZ DEFAULT now(),
    checksum    TEXT,                    -- sha256 for re-index detection
    UNIQUE(path, repo)
);
```

### `chunks` table — doc chunk registry (vector-side)
```sql
CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    file_id     INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    level       TEXT NOT NULL,           -- 'parent' | 'child'
    line_start  INTEGER,
    line_end    INTEGER,
    qdrant_id   UUID NOT NULL            -- Qdrant point ID
);
```

### `graph_nodes` table — code node registry (graph-side)
```sql
CREATE TABLE graph_nodes (
    id          SERIAL PRIMARY KEY,
    file_id     INTEGER REFERENCES files(id) ON DELETE CASCADE,
    node_type   TEXT NOT NULL,           -- 'function' | 'class' | 'module' | 'file'
    name        TEXT NOT NULL,
    line_start  INTEGER,
    line_end    INTEGER,
    falkordb_id TEXT NOT NULL            -- FalkorDB node ID
);
```

---

## Ingestion Pipeline

### 1. Document Ingestion (Docs + Code-in-Docs → Qdrant)

#### Step 1 — File registration
- Hash file, check `files` table for existing checksum
- If unchanged, skip. Otherwise insert/update row, get `file_id`
- Set `source_type = 'doc'`

#### Step 2 — Hierarchical chunking
- Chunk document into **parent chunks** (~512-1024 tokens, respecting semantic boundaries — headings, code blocks, sections)
- Each parent chunk is further split into **child chunks** (~128-256 tokens)
- Chunker must keep code blocks intact — never split mid-code-block
- Store parent/child relationship in `chunks` table with `line_start`, `line_end`

#### Step 3 — Parent chunk summarization
- For each parent chunk, call LLM to generate a structured summary
- Summary format for **code-heavy chunks** (structured):
  ```
  Exports: X, Y, Z
  Depends on: A, B
  Key types: Foo, Bar
  Purpose: <1-2 sentence description>
  ```
- Summary format for **prose-heavy chunks** (natural language):
  ```
  <2-3 sentence abstractive summary>
  ```
- Detect chunk type heuristically (code token ratio threshold)

#### Step 4 — Embedding
- Embed **parent chunk summaries** using `nomic-embed-text` (local) or `text-embedding-3-large` (API)
- Embed **child chunks** using same model
- For child chunks containing code, use `nomic-embed-code` or `voyage-code-3`
- Store summary vectors and child chunk vectors separately in Qdrant as distinct named collections or payload-flagged

#### Step 5 — Qdrant upsert
- Upsert parent summary vectors into `summaries` collection
  - Payload: `{ file_id, chunk_id, line_start, line_end, level: "summary" }`
- Upsert child chunk vectors into `chunks` collection
  - Payload: `{ file_id, chunk_id, parent_chunk_id, line_start, line_end, level: "child" }`
- Store Qdrant point UUIDs back into `chunks` table (`qdrant_id`)

---

### 2. Codebase Ingestion (Codebases → FalkorDB)

#### Step 1 — File registration
- Register each source file in `files` table, `source_type = 'codebase'`

#### Step 2 — AST extraction (Tree-sitter)
For each file extract:
- **Functions**: name, parameters, return type, line range, docstring
- **Classes**: name, methods, base classes, line range
- **Imports**: resolved module paths
- **Module-level variables / constants**

#### Step 3 — Call graph generation
- Run static call graph analyzer per language:
  - Python: `pyan3 --dot` → parse DOT output → edges
  - Go: `golang.org/x/tools/go/callgraph` (RTA algorithm)
  - JS/TS: `ts-morph` AST traversal
  - Rust: `cargo-call-stack` or custom `syn` traversal
- Output: list of `(caller_qualified_name, callee_qualified_name)` tuples
- Map qualified names back to AST-extracted node IDs

#### Step 4 — FalkorDB graph construction

**Node types:**
```cypher
(:File   { file_id, path, language, repo })
(:Module { file_id, name, path })
(:Class  { file_id, name, line_start, line_end, embedding: float[] })
(:Function { file_id, name, line_start, line_end, signature, docstring, embedding: float[] })
```

**Edge types:**
```cypher
(:File)-[:CONTAINS]->(:Function)
(:File)-[:CONTAINS]->(:Class)
(:Class)-[:HAS_METHOD]->(:Function)
(:Function)-[:CALLS]->(:Function)
(:Module)-[:IMPORTS]->(:Module)
(:Class)-[:INHERITS]->(:Class)
```

#### Step 5 — Code embedding on nodes
- Embed each `Function` and `Class` node using `voyage-code-3` (API) or `nomic-embed-code` (local)
- Embedding input: `signature + docstring + first N lines of body`
- Store embedding directly on node as `embedding` property (FalkorDB supports vector properties)
- Register node in `graph_nodes` table with `falkordb_id`

---

## Qdrant Collections Schema

### `summaries` collection
- Vector: dense (1536d or model-dependent)
- Sparse: SPLADE or BM25
- Late interaction: ColBERT-style multi-vector
- Payload fields: `file_id`, `chunk_id`, `line_start`, `line_end`, `source_type`

### `chunks` collection
- Same vector config as summaries
- Payload fields: `file_id`, `chunk_id`, `parent_chunk_id`, `line_start`, `line_end`, `source_type`

---

## Query Pipeline

### Phase 1 — Query expansion
- User query → LLM generates:
  1. **Rewritten query**: cleaned, deambiguated single query for direct retrieval
  2. **Checklist**: 4-5 structured sub-questions covering all query nuance
  
  Output schema (strict JSON via tool calling):
  ```json
  {
    "rewritten_query": "...",
    "checklist": [
      { "id": 1, "question": "...", "answered": false },
      ...
    ]
  }
  ```

### Phase 2 — Initial retrieval (parallel)

Run **simultaneously**:

**A) Summary search** (Qdrant `summaries` collection)
- RRF over dense + sparse + late interaction vectors of `rewritten_query`
- Return top-K summary results → extract `file_id` list

**B) Direct vector search** (Qdrant `chunks` collection)  
- Same RRF pipeline on `rewritten_query` against child chunks
- Return top-K chunk results → extract `file_id` list (safety net for summary misses)

Merge and deduplicate `file_id` sets from A and B.

### Phase 3 — Agentic retrieval loop

#### Subagent input schema:
```json
{
  "user_query": "...",
  "checklist": [...],
  "accumulated_context": [...],
  "iteration": 1
}
```

#### Subagent output schema (strict JSON via tool calling):
```json
{
  "checklist": [
    { "id": 1, "question": "...", "answered": true },
    { "id": 2, "question": "...", "answered": false },
    ...
  ],
  "all_answered": false,
  "retrieval_queries": [
    {
      "query": "...",
      "target": "vector" | "graph" | "both",
      "filters": {
        "file_ids": [42, 87],
        "node_type": "function" | "class" | null,
        "language": "python" | null
      }
    }
  ]
}
```

#### Loop logic:
```
iteration = 1
max_iterations = 5
accumulated_context = []

while iteration <= max_iterations:
    subagent = new SubAgent(
        user_query,
        checklist,
        accumulated_context,
        iteration
    )
    result = subagent.run()

    if result.all_answered:
        break

    for rq in result.retrieval_queries:
        if rq.target in ["vector", "both"]:
            chunks = qdrant_retrieve(rq.query, rq.filters)
        if rq.target in ["graph", "both"]:
            nodes = falkordb_retrieve(rq.query, rq.filters)
        new_context = rerank(chunks + nodes, rq.query)
        accumulated_context = deduplicate(accumulated_context + new_context)

    checklist = result.checklist
    iteration += 1
```

#### Subagent fresh context design:
- Each subagent receives only: original user query + current checklist state + accumulated context (compressed to fit context window)
- Subagent dies after one turn — no conversation history carried forward
- Accumulated context passed as structured list of `{ file_id, path, line_start, line_end, content }` objects, not raw text

### Phase 4 — Final rerank
- Pass all accumulated context chunks/nodes through FlashRerank against original user query
- Top-N results selected for LLM context window

### Phase 5 — Generation
- LLM receives: original query + reranked context + citation metadata
- Streams response tokens via SSE
- Citations emitted as structured SSE events alongside token stream:
  ```json
  { "type": "citation", "file_id": 42, "path": "src/auth/middleware.go", "line_start": 87, "line_end": 112 }
  { "type": "token", "content": "The JWT expiry..." }
  ```

---

## Retrieval Implementations

### Qdrant RRF Retrieval

```python
async def vector_retrieve(query: str, filters: dict) -> list[ChunkResult]:
    dense_results  = await qdrant.search(collection, dense_vector(query),   filter=filters, limit=20)
    sparse_results = await qdrant.search(collection, sparse_vector(query),  filter=filters, limit=20)
    late_results   = await qdrant.search(collection, late_vector(query),    filter=filters, limit=20)
    return rrf_merge([dense_results, sparse_results, late_results])
```

- `file_ids` filter maps to Qdrant payload filter on `file_id` field
- RRF constant k=60 (standard)

### FalkorDB Retrieval

Two modes emitted by subagent:

**Semantic (vector on nodes):**
```cypher
MATCH (f:Function)
WHERE f.file_id IN $file_ids
RETURN f, vector_distance(f.embedding, $query_embedding) AS score
ORDER BY score ASC
LIMIT 20
```

**Structural traversal:**
```cypher
-- callers of a function
MATCH (caller:Function)-[:CALLS]->(f:Function {name: $name})
RETURN caller

-- full call chain to depth N
MATCH path = (f:Function {name: $name})-[:CALLS*1..5]->(dep)
RETURN path

-- all symbols in a file
MATCH (file:File {file_id: $file_id})-[:CONTAINS]->(node)
RETURN node

-- inheritance chain
MATCH path = (c:Class {name: $name})-[:INHERITS*1..10]->(base)
RETURN path
```

The subagent selects which query type to emit based on whether the query is semantic ("find functions related to auth") vs structural ("what calls handleJWT").

---

## Model Registry (Persistent Workers)

All models loaded once at startup via Taskiq actor pattern — never cold-loaded per request:

```python
class ModelRegistry:
    embedder_local: SentenceTransformer      # nomic-embed-text / nomic-embed-code
    embedder_code_api: VoyageEmbedder        # voyage-code-3
    embedder_text_api: OpenAIEmbedder        # text-embedding-3-large
    reranker: FlashRerank                    # ms-marco or bge-reranker-v2-m3
    llm_local: OllamaClient                  # user-configured model
    llm_api: AnthropicClient | OpenAIClient  # user-configured
    sparse_encoder: SpladeEncoder            # for sparse vectors
    late_encoder: ColBERTEncoder             # for late interaction vectors
```

User config selects which embedder/LLM is used per operation type (query expansion, subagent, generation, classification).

---

## FastAPI SSE Streaming

### Endpoint: `POST /query`

```python
@router.post("/query")
async def query(request: QueryRequest):
    return EventSourceResponse(query_stream(request))

async def query_stream(request: QueryRequest):
    # Phase 1 — expansion
    yield sse_event("status", {"phase": "expanding", "message": "Expanding query..."})
    expansion = await expander.expand(request.query)
    yield sse_event("expansion", expansion.dict())

    # Phase 2 — initial retrieval
    yield sse_event("status", {"phase": "retrieving", "message": "Searching summaries..."})
    file_ids = await initial_retrieve(expansion.rewritten_query)
    yield sse_event("file_ids", {"count": len(file_ids)})

    # Phase 3 — agentic loop
    async for loop_event in agentic_loop(request.query, expansion, file_ids):
        yield sse_event(loop_event.type, loop_event.data)

    # Phase 4 — generation
    async for token in llm.stream(context, request.query):
        yield sse_event("token", {"content": token})

    yield sse_event("done", {})
```

SSE event types emitted:
- `status` — phase label + human message for TUI progress panel
- `expansion` — checklist items for TUI to render
- `file_ids` — initial file scope count
- `checklist_update` — per subagent iteration, which items got checked off
- `iteration` — current loop iteration number
- `citation` — `{ file_id, path, line_start, line_end }`
- `token` — LLM generation token
- `done` — stream complete

---

## TUI Architecture (Bubbletea)

### Root Model (`app.go`)
Composes all sub-models. Handles SSE stream lifecycle.

### SSE Stream → Bubbletea (`stream.go`)
```go
func listenSSE(url string) tea.Cmd {
    return func() tea.Msg {
        resp, _ := http.Post(url, "application/json", body)
        reader := bufio.NewReader(resp.Body)
        for {
            line, _ := reader.ReadString('\n')
            event := parseSSELine(line)
            // dispatch as tea.Msg to root model
        }
    }
}
```

Each SSE event type maps to a distinct `tea.Msg` type dispatched into the Bubbletea update loop.

### Panels

**Query input panel** (`query.go`)
- Single-line input with history (↑/↓)
- Mode indicator: `[docs]` / `[code]` / `[auto]`

**Progress panel** (`progress.go`)
- Live checklist: renders 4-5 sub-questions with `[ ]` / `[✓]` per item
- Current iteration counter
- Active retrieval target (`→ vector` / `→ graph` / `→ both`)
- Phase label from `status` events

**Results panel** (`results.go`)
- Streaming LLM output with word-wrap
- Syntax highlighting for code blocks in response

**Sources panel** (`sources.go`)
- List of citations: `path/to/file.go:87-112`
- Populated incrementally as `citation` events arrive during generation
- Selectable — pressing enter on a citation opens file at line in `$EDITOR`

### Key Bindings
```
/        — focus query input
Tab      — cycle panels
↑↓       — scroll active panel
Enter    — on citation: open in $EDITOR
Ctrl+C   — cancel active stream
q        — quit (when not in input mode)
?        — help overlay
i        — index current directory
```

---


## Configuration (`config.toml`)
***THIS WOULD BE A TOML FILE NOT YAML***

```yaml
models:
  embedding_local: nomic-embed-text       # Ollama model name
  embedding_code_local: nomic-embed-code
  embedding_api: voyage-code-3            # Used for code node indexing
  llm_local: llama3.1:8b                  # Ollama model name
  llm_api: claude-sonnet-4-20250514
  reranker: ms-marco-MiniLM-L-6-v2

  # per-operation model selection: 'local' | 'api'
  ops:
    query_expansion: local
    subagent: local
    generation: api
    classifier: local

databases:
  postgres: postgresql://localhost:5432/ragtui
  qdrant: http://localhost:6333
  falkordb:
    host: localhost
    port: 6380                            # remapped from 6379

ingestion:
  chunk_size_parent: 768
  chunk_size_child: 192
  summary_model: local                    # which LLM for summarization
  code_embedding_threshold: 0.4          # ratio of code tokens to use code embedder

retrieval:
  initial_summary_top_k: 20
  initial_direct_top_k: 10
  rerank_top_n: 15
  max_iterations: 5
  checklist_size: 5

api:
  host: 0.0.0.0
  port: 8000
```

---

## Ingestion CLI Commands (TUI keybinds / slash commands)

```
:index docs <path>       — index docs directory into Qdrant
:index code <path>       — index codebase into FalkorDB
:index all <path>        — index both (auto-detect by file type)
:reindex docs <path>     — full re-index docs (drop + rebuild)
:reindex code <path>     — full re-index codebase (drop + rebuild)
:status                  — show index stats (file count, node count, chunk count)
```


---

## Implementation Order

1. **PostgreSQL schema + migrations** — file registry foundation everything else depends on
2. **Model registry** — persistent embedder/reranker workers via Taskiq
3. **Hierarchical chunker** — parent/child splitting with code block preservation
4. **Qdrant ingestor** — summary + chunk upsert with RRF collection setup
5. **Vector retriever** — RRF pipeline (dense + sparse + late) + FlashRerank
6. **AST parser** — Tree-sitter extraction for target languages
7. **Call graph generator** — per-language static analysis → edge list
8. **FalkorDB ingestor** — node/edge upsert, code embeddings on nodes
9. **Graph retriever** — semantic vector search + Cypher traversal queries
10. **Query expander** — rewritten query + checklist generation
11. **Subagent** — single iteration: checklist assessment + targeted query emission
12. **Agentic loop** — orchestration, context accumulation, deduplication
13. **FastAPI SSE endpoints** — streaming pipeline assembly
14. **Bubbletea TUI** — panels, SSE stream reader, key bindings
15. **Config system** — yaml config + per-operation model selection
16. **CLI ingestion commands** — TUI slash commands + index management
