
███████╗ ██████╗ ███████╗██████╗  █████╗ 
██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔══██╗
█████╗  ██║   ██║███████╗██████╔╝███████║
██╔══╝  ██║   ██║╚════██║██╔══██╗██╔══██║
██║     ╚██████╔╝███████║██║  ██║██║  ██║
╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
**F**lexible **O**pen **S**ource **R**etrieval **A**ugmented Generation System

> !! **Pre-Alpha Stage**: Currently backend-focused with un-styled working interface. Not yet MVP.

---
╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯

## Vision

FOSRA is a next-gen RAG and internet search interface built for **maximum interoperability**. FOSRA lets you choose, customize, and swap every component of your RAG pipeline—from document ingestion to final output generation.

**Core Philosophy**: Intelligence through modularity. Every step of retrieval, search, and output generation is transparent, configurable, and optimized.

---
╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯
## Architecture

### Deployment Modes
- **Docker**: Containerized deployment for server environments
- **Tauri**: Native desktop application wrapper

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.13+, FastAPI, Async-first (uvloop) |
| **Frontend** | React 19, TypeScript, Tailwind CSS 4, Vite  |
| **State** | Zustand, TanStack Query + Router |
| **UI** | Radix UI primitives, shadcn/ui patterns |
| **Vector DB** | Qdrant, Elasticsearch |
| **Processing** | Docling, Unstructured, Chonkie |
| **LLM Gateway** | LiteLLM (universal model access) |
| **Observability** | Logfire, OpenTelemetry |

╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯

---

## Key Features (In Development)

### **Modular Pipeline**
- **Ingestion**: Pluggable document processors (PDF, Office, web, media)
- **Chunking**: Multiple strategies (semantic, fixed, hierarchical)
- **Embedding**: FastEmbed, Sentence Transformers, or remote APIs
- **Retrieval**: Vector search, keyword, hybrid, reranking
- **Generation**: Any LLM via LiteLLM (OpenAI, Anthropic, local, etc.)

### **Intelligent Retrieval**
- Context-aware search routing
- Automatic query decomposition and Expansion
- Multi-hop reasoning support
- Dynamic reranking (FlashRank, cross-encoders)

### **Internet Search Integration**
- Firecrawl, Tavily, Exa, Linkup + More
- GitHub code search
- Custom Search

### **Advanced Capabilities**
- Async-first architecture throughout
- Distributed task processing (Celery/Taskiq)
- Real-time streaming responses
- Full observability with tracing
- Agentic workflows (LangGraph, DeepAgents)

---
╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯

## Project Structure

```
fosra/
├── backend/
│   ├── src/api/           # FastAPI routes & schemas
│   ├── src/domain/        # Business logic & exceptions
│   ├── src/services/      # Core services (conversation, retrieval, processing, workspace)
│   ├── src/storage/       # Repositories & storage utils
│   ├── src/tasks/         # Background task handlers
│   ├── src/settings/      # Configuration schemas
│   ├── migrations/        # Alembic database migrations
│   └── tests/            # Test suites
└── frontend/
    ├── assets/           # Static resources
    ├── components/       # React components (ui, fosra-ui, ai-elements, schemas)
    ├── hooks/            # Custom React hooks
    ├── lib/api/          # Generated API client (@tanstack, core, client)
    └── routes/           # TanStack Router routes
```
```
```
```
```
---
╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯

## Development Status

### Implemented
- Core async backend infrastructure
- Document parsing (Docling, Unstructured)
- Vector database abstractions (Qdrant, Elasticsearch)
- API endpoints (OpenAPI-generated)
- Minimal frontend interface


### In Progress
- Pipeline orchestration UI 
- Search provider integrations 
- Docker production builds

### Planned
- Multi-tenant architecture
- Advanced agent workflows
- Collaborative features
- Plugin marketplace
- Tauri desktop wrapper

---
╭─────────────────────────────────────────────────────────────────╮
╰─────────────────────────────────────────────────────────────────╯

## Quick Start

> **Note**: Pre-alpha setup requires manual configuration.
> **Note**: No Compose Yet: Docker Containers for Qdrant,Postgres,Redis needed

### Backend
```bash
cd backend
uv sync
fastapi dev src/main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/FOSRA
QDRANT_URL=http://localhost:6333
LITELLM_API_KEY=your_key

# Optional providers
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
FIRECRAWL_API_KEY=...
```

---

## Dependencies Highlights

### Backend (Python)
- **FastAPI** + **uvicorn**: High-performance async API
- **LangChain/LangGraph**: LLM orchestration and agent workflows
- **Docling + Unstructured**: Enterprise-grade document parsing
- **Chonkie**: Intelligent chunking with semantic boundaries
- **FastEmbed**: Efficient local embeddings
- **Taskiq**: Distributed task queues

### Frontend (TypeScript/React)
- **TanStack ecosystem**: Query, Router, Form, Virtual
- **Radix UI**: Accessible, unstyled primitives
- **Zustand**: Lightweight state management
- **AI SDK**: Streaming LLM interactions
- **React Hook Form + Zod**: Type-safe forms

---

## Roadmap

| Phase | Target | Goals |
|-------|--------|-------|
| **Alpha** | Q1 2026 | MVP with core RAG, basic UI, Docker |
| **Beta** | Q2 2026 | Tauri app, advanced search, plugins |
| **1.0** | Q3 2026 | Production-ready, marketplace, cloud |

---
## Contributing

FOSRA is in early development. Contributions, feedback, and issue reports are welcome as the architecture is shaped.

---

## License

[License TBD - likely Apache 2.0 or MIT]

---

