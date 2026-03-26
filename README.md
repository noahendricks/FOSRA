                                
 <pre>
<p align="center">
███████╗ ██████╗ ███████╗██████╗  █████╗ 
██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔══██╗
███████╗██║   ██║███████╗██████╔╝███████║
██╔════╝██║   ██║╚════██║██╔══██╗██╔══██║
██║     ╚██████╔╝███████║██║  ██║██║  ██║
╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
<b>F</b>lexible <b>O</b>pen <b>S</b>ource <b>R</b>etrieval <b>A</b>ugmented Generation System
</pre>
</p>

<p align="center">
◆ ◆ ◆
</p>

<p align="center">
<b>Pre-Alpha Stage</b>: Currently backend-focused with un-styled working interface. Not yet MVP.
</p>

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>
<p align="center">
FOSRA is a next-gen RAG and internet search interface built for <b>maximum modularity</b>.<br>
FOSRA lets you choose, customize, and swap every component of your RAG pipeline—<br>
from document ingestion to final output generation.
</p>

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>

<p align="center">
<b>Architecture</b>
</p>

<p align="center">
<b>Deployment Modes</b><br>
Docker • Containerized deployment for server environments<br>
Tauri • Native desktop application wrapper
</p>

<p align="center">
  <b>Tech Stack</b>
</p>

<table align="center">
  <thead>
    <tr>
      <th align="center">Layer</th>
      <th align="center">Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Backend</td>
      <td align="center">Python 3.13+, FastAPI, Async-first (uvloop)</td>
    </tr>
    <tr>
      <td align="center">Frontend</td>
      <td align="center">React 19, TypeScript, Tailwind CSS 4, Vite</td>
    </tr>
    <tr>
      <td align="center">State</td>
      <td align="center">Zustand, TanStack Query + Router</td>
    </tr>
    <tr>
      <td align="center">UI</td>
      <td align="center">Radix UI primitives, shadcn/ui patterns</td>
    </tr>
    <tr>
      <td align="center">Vector DB</td>
      <td align="center">Qdrant, Elasticsearch</td>
    </tr>
    <tr>
      <td align="center">Processing</td>
      <td align="center">Docling, Unstructured, Chonkie</td>
    </tr>
    <tr>
      <td align="center">LLM Gateway</td>
      <td align="center">LiteLLM (universal model access)</td>
    </tr>
    <tr>
      <td align="center">Observability</td>
      <td align="center">Logfire, OpenTelemetry</td>
    </tr>
  </tbody>
</table>

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>Key Features (In Development)</b>
</p>

<p align="center">
<b>◆ Modular Pipeline </b>
</p>

<p align="center">
Ingestion • Pluggable document processors (PDF, Office, web, media)<br>
Chunking • Multiple strategies (semantic, fixed, hierarchical)<br>
Embedding • FastEmbed, Sentence Transformers, or remote APIs<br>
Retrieval • Vector search, keyword, hybrid, reranking<br>
Generation • Any LLM via LiteLLM (OpenAI, Anthropic, local, etc.)
</p>

<p align="center">
<b>◆ Intelligent Retrieval </b>
</p>

<p align="center">
Context-aware search routing<br>
Automatic query decomposition and Expansion<br>
Multi-hop reasoning support<br>
Dynamic reranking (FlashRank, cross-encoders)
</p>

<p align="center">
<b> Internet Search Integration ◆</b>
</p>

<p align="center">
Firecrawl • Tavily • Exa • Linkup • Custom Search • GitHub code search
</p>

<p align="center">
<b>Advanced Capabilities ◆</b>
</p>

<p align="center">
Async-first architecture throughout<br>
Distributed task processing (Celery/Taskiq)<br>
Real-time streaming responses<br>
Full observability with tracing<br>
Agentic workflows (LangGraph, DeepAgents)
</p>


<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>

<p align="center">
<b>Project Structure</b>
</p>

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


<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>Development Status</b>
</p>

<p align="center">
<b>[ Implemented ]</b>
</p>

<p align="center">
Core async backend infrastructure<br>
Document parsing (Docling, Unstructured)<br>
Vector database abstractions (Qdrant, Elasticsearch)<br>
API endpoints (OpenAPI-generated)<br>
Minimal frontend interface
</p>

<p align="center">
<b>[ In Progress ]</b>
</p>

<p align="center">
Pipeline orchestration UI<br>
Search provider integrations<br>
Docker production builds
</p>

<p align="center">
<b>[ Planned ]</b>
</p>

<p align="center">
Multi-tenant architecture • Advanced agent workflows<br>
Collaborative features • Plugin marketplace • Tauri desktop wrapper
</p>


<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>Quick Start</b>
</p>

<p align="center">
<i>Note: Pre-alpha setup requires manual configuration.</i><br>
<i>No Compose Yet: Docker Containers for Qdrant, Postgres, Redis needed</i>
</p>

<p align="center">
<b>Backend</b>
</p>

```bash
cd backend
uv sync
fastapi dev src/main.py
```

<p align="center">
<b>Frontend</b>
</p>

```bash
cd frontend
npm install
npm run dev
```

<p align="center">
<b>Environment Variables</b>
</p>

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

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>Dependencies Highlights</b>
</p>

<p align="center">
<b>Backend (Python)</b>
</p>

<p align="center">
FastAPI + uvicorn • High-performance async API<br>
LangChain/LangGraph • LLM orchestration and agent workflows<br>
Docling + Unstructured • Enterprise-grade document parsing<br>
Chonkie • Intelligent chunking with semantic boundaries<br>
FastEmbed • Efficient local embeddings<br>
Taskiq • Distributed task queues
</p>

<p align="center">
<b>Frontend (TypeScript/React)</b>
</p>

<p align="center">
TanStack ecosystem • Query, Router, Form, Virtual<br>
Radix UI • Accessible, unstyled primitives<br>
Zustand • Lightweight state management<br>
AI SDK • Streaming LLM interactions<br>
React Hook Form + Zod • Type-safe forms
</p>

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>

<p align="center">
  <b>Roadmap</b>
</p>

<table align="center">
  <thead>
    <tr>
      <th align="center">Phase</th>
      <th align="center">Target</th>
      <th align="center">Goals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Alpha</td>
      <td align="center">Q1 2026</td>
      <td align="center">MVP with full RAG and search, UI, Docker</td>
    </tr>
    <tr>
      <td align="center">Beta</td>
      <td align="center">Q2 2026</td>
      <td align="center">Tauri app, advanced search, plugins</td>
    </tr>
    <tr>
      <td align="center">1.0</td>
      <td align="center">Q3 2026</td>
      <td align="center">Production-ready, cloud / domain deployment</td>
    </tr>
  </tbody>
</table>

<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>Contributing</b>
</p>

<p align="center">
FOSRA is in early development. Contributions, feedback, and issue reports<br>
are welcome as the architecture is shaped.
</p>


<p align="center">
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
</p>


<p align="center">
<b>License</b>
</p>

<p align="center">
[License TBD - likely Apache 2.0 or MIT]
</p>

<p align="center">
◆ ◆ ◆
</p>

---

