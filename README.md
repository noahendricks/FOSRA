<p align="center">
  <img src="static/fosra-title.svg" alt="FOSRA" width="800">
</p>

<p align="center">
  <img src="static/animation.gif" width="500">
</p>

<p align="center">
  <code>v0.2.0</code> -- Alpha
</p>


---

FOSRA is a modular RAG system and internet search interface. Every component of the pipeline — document parsing, chunking, embedding, retrieval, reranking, generation — is swappable. Run it with local models, cloud APIs, or both.

The backend is a Python async-first FastAPI service. The interface is a terminal UI built on OpenTUI and SolidJS, running on Bun. There is no web frontend.

---


## Features

### Modular Pipeline

*Completely Configurable* — Every stage is swappable: parsing (Docling, Unstructured, Kreuzberg, LlamaParse), chunking (Chonkie, HiChunk), embedding (FastEmbed, Sentence Transformers, Voyage), retrieval (vector, graph, hybrid), reranking (FlashRank, BGE v2), and generation (any model via LiteLLM).


### Intelligent Retrieval

- Context-aware search routing
- Automatic query decomposition and expansion
- Iterative multi-hop retrieval with configurable depth
- Dynamic reranking with FlashRank and cross-encoders

### Internet Search

Built-in search provider integrations: Firecrawl, Tavily, Exa, Linkup, GitHub code search.

### Session Management

- Persistent conversation history (PostgreSQL)
- Agent runner with subagent delegation
- Streaming responses via SSE
- LangGraph and DeepAgents integration for complex workflows



## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Bun](https://bun.sh/) (JavaScript runtime)
- PostgreSQL
- Qdrant
- FalkorDB (Redis-compatible, port 6379)

### Backend

```bash
cd backend
uv sync
cp .env.example .env          # add your API keys
fastapi dev src/app.py
```

### Terminal UI

```bash
cd tui
bun install
bun run dev
```

### Environment Variables

```bash
# REQUIRED
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/fosra
QDRANT_URL=http://localhost:6333

# LLM PROVIDERS (at least one)
LITELLM_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# SEARCH PROVIDERS (optional)
TAVILY_API_KEY=...
FIRECRAWL_API_KEY=...
EXA_API_KEY=...
LINKUP_API_KEY=...
```

---

## Configuration

All runtime behavior is controlled through `config.toml`. Secrets go in environment variables.

```toml
[models]
embedding_local = "nomic-ai/nomic-embed-text-v1.5"
embedding_api = "voyage-code-3"
llm_local = "llama3.1:8b"
llm_api = "claude-sonnet-4-20250514"
reranker = "BAAI/bge-reranker-v2-m3"

[models.ops]
query_expansion = "local"      # which model handles each operation
subagent = "local"
generation = "api"
classifier = "local"
summarization = "local"

[databases]
postgres_url = "postgresql+asyncpg://localhost:5432/fosra"
qdrant_url = "http://localhost:6333"

[databases.falkordb]
host = "localhost"
port = 6379
graph_name = "fosra"

[ingestion]
chunk_size_parent = 768
chunk_size_child = 192

[retrieval]
initial_summary_top_k = 20
initial_direct_top_k = 10
rerank_top_n = 15
max_iterations = 5
```

You can route each operation independently — run generation on Claude while keeping query expansion and classification on a local Llama model.

---
## Development

### Project Structure

```
fosra/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── oc/            # session operations (messages, files, shell, permissions)
│   │   │   │   ├── tui.py         # terminal UI endpoints
│   │   │   │   ├── ingestion.py   # document ingestion
│   │   │   │   └── workspace.py   # workspace management
│   │   │   ├── schemas/           # request/response models
│   │   │   ├── dependencies.py    # FastAPI dependency injection
│   │   │   └── exception_handlers.py
│   │   ├── services/
│   │   │   ├── session/           # conversation service, agent runner, subagent,
│   │   │   │                      # retrieval pipeline, query expander, event emitter
│   │   │   ├── processing/        # document loaders, chunkers, embedders, callgraph
│   │   │   ├── retrieval/         # vector search, graph retrieval, reranking
│   │   │   └── model_registry.py  # model configuration
│   │   ├── storage/               # repositories, data models
│   │   ├── domain/                # business logic, exceptions
│   │   ├── settings/              # config loader (config.toml)
│   │   ├── tasks/                 # background job handlers
│   │   └── migrations/            # Alembic database migrations
│   └── tests/
├── tui/
│   ├── src/
│   │   ├── fosra/
│   │   │   └── client/            # HTTP/SSE client for backend communication
│   │   ├── component/             # UI components
│   │   ├── routes/                # terminal UI routing
│   │   ├── store/                 # state management
│   │   ├── context/               # SolidJS context providers
│   │   ├── ui/                    # base UI primitives
│   │   └── util/                  # shared utilities
│   └── package.json
├── config.toml                    # runtime configuration (models, databases, retrieval)
└── pyproject.toml
```

---

## Architecture

<div align="center">

  ## Tech Stack
  
  | Layer | Technology |
  |-------|-----------|
  | Backend | Python 3.12+, FastAPI, uvicorn, uvloop |
  | Terminal UI | OpenTUI 0.1.86, SolidJS, Bun, TypeScript |
  | Vector DB | Qdrant |
  | Graph DB | FalkorDB |
  | Relational DB | PostgreSQL (asyncpg, SQLAlchemy) |
  | LLM Gateway | LiteLLM (universal model routing) |
  | Task Queue | Taskiq |
  | Observability | OpenTelemetry, Loguru, structlog |
  | Migrations | Alembic |
  
  ---

```
+--------------------------------------+
|            Terminal UI                |
|       OpenTUI + SolidJS (Bun)        |
+------------------+-------------------+
                   | HTTP / SSE
+------------------v-------------------+
|          FastAPI Backend              |
|       Python 3.13+ / uvloop          |
|                                      |
|  +--------------+ +----------------+ |
|  |  Ingestion   | |    Session     | |
|  |  Pipeline    | |    Manager     | |
|  +------+-------+ +-------+-------+ |
|         |                  |         |
|  +------v-------+ +-------v-------+ |
|  |  Processing  | |   Retrieval   | |
|  |  Services    | |   Pipeline    | |
|  +------+-------+ +-------+-------+ |
|         |                  |         |
+---------v------------------v---------+
|       Storage + External Services    |
+--------------------------------------+
  PostgreSQL   Qdrant     FalkorDB
  LiteLLM      Taskiq     Search APIs
```

</div>

Special thanks and obligatory shoutout to the OpenCode team for the TUI ! ✨

---

## Contributing

FOSRA is in active development. The architecture is settling but not frozen — contributions, bug reports, and feedback are welcome.

---

## License

Apache 2.0
