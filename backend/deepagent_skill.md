# DeepAgents — SKILL.md

***DISCLAIMER: THIS IS A SKILL MD FOR DEEPAGENTS, BUT IF YOU NEED ANY DOUBLE CHECKING USE CONTEXT7 MCP FOR MORE INFORMATION ON WHAT YOU ARE SEEKING***



## Overview

`deepagents` is a LangChain package that implements the Claude Code / Deep Research agent architecture: a planning tool, file system tools, subagents, and conversation summarization, composable via middleware. It wraps LangGraph and returns a `CompiledStateGraph`.

```bash
pip install deepagents
# or
uv add deepagents
```

---

## 1. create_deep_agent — Main Entry Point

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[my_tool_1, my_tool_2],
    system_prompt="You are a coding assistant with access to retrieval tools.",
    middleware=[],         # additional middleware after built-in stack
    subagents=None,        # no sub-agents — flat tool list for 9B models
    backend=None,          # defaults to StateBackend (ephemeral)
    checkpointer=None,     # pass LangGraph checkpointer for persistence
    store=None,            # pass LangGraph BaseStore if using StoreBackend
    debug=False,
)
```

### Full signature

```python
create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph
```

**Returns**: A compiled LangGraph `CompiledStateGraph`. Invoke with `.invoke()` or `.astream()`.

### Built-in middleware stack (always applied, in order)

1. `TodoListMiddleware` — `write_todos` tool for planning
2. `FilesystemMiddleware` — `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, optionally `execute`
3. `SubAgentMiddleware` — `task` tool (only active if `subagents` is provided)
4. `SummarizationMiddleware` — automatic conversation compaction
5. `AnthropicPromptCachingMiddleware` — prompt caching for Anthropic models
6. `PatchToolCallsMiddleware` — repairs dangling tool calls

Custom middleware passed via `middleware=` is appended **after** this stack.

---

## 2. Invocation

### Synchronous

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "How does the DataProcessor class parse CSV files?"}]
})
print(result["messages"][-1].content)
```

### Asynchronous streaming (preferred)

```python
async for event in agent.astream(
    {"messages": [{"role": "user", "content": "How does setQueryData work?"}]},
    config={"configurable": {"thread_id": session_id}},  # for persistence
    stream_mode="values",
):
    last_msg = event["messages"][-1]
    if hasattr(last_msg, "content"):
        print(last_msg.content, end="", flush=True)
```

### With conversation history

```python
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "explain QueryClient"},
        {"role": "assistant", "content": "QueryClient is the core class..."},
        {"role": "user", "content": "what prefetch methods does it have?"},
    ]
})
```

---

## 3. Tools

Tools passed to `create_deep_agent` are appended to the built-in tool set. They must be `BaseTool`, callables decorated with `@tool`, or dicts.

```python
from langchain_core.tools import tool

@tool
def run_retrieval_pipeline(query: str) -> str:
    """Search the codebase and documentation for information relevant to the query.
    Use this for any question about code structure, API usage, or documentation.
    
    Args:
        query: The natural language question to answer from the codebase.
    """
    # calls inner LangGraph pipeline
    result = retrieval_pipeline.invoke({"user_query": query})
    return result["context"]

@tool
def read_project_file(path: str) -> str:
    """Read the contents of a file in the current project.
    
    Args:
        path: Relative path from project root.
    """
    return open(f"{PROJECT_ROOT}/{path}").read()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[run_retrieval_pipeline, read_project_file],
    system_prompt=AGENT_SYSTEM_PROMPT,
)
```

**RULE for FOSRA**: Use a flat tool list. Never use `subagents` with a 9B model — each delegation hop is a reasoning failure point.

---

## 4. Middleware

### FilesystemMiddleware

Controls file access, tool eviction, and optional shell execution.

```python
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend, StateBackend, CompositeBackend

# Read-only filesystem (for FOSRA — no writes allowed)
fs_middleware = FilesystemMiddleware(
    backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual=False),
    tool_token_limit_before_evict=20000,  # evict large results to file
)

# Ephemeral in-memory (default, no disk access)
fs_middleware = FilesystemMiddleware(backend=StateBackend())

# Hybrid: ephemeral default + persistent /memories/ path
backend = CompositeBackend(
    default=StateBackend(),
    routes={"/memories/": StoreBackend()}
)
fs_middleware = FilesystemMiddleware(backend=backend)
```

**`tool_token_limit_before_evict`**: When a tool result exceeds this token count, it is written to the backend and replaced in the conversation with a file reference + truncated preview. Prevents context window saturation from large retrievals.

### SummarizationMiddleware

Built-in automatic compaction. Triggers when conversation approaches context limit. No configuration needed in most cases — it is included in the default stack.

For manual compaction control:

```python
from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)

# Auto-summarization with custom settings
summarization = create_summarization_middleware(
    model="anthropic:claude-haiku-4-5-20251001",  # cheaper model for summarization
    token_threshold=0.8,    # compact at 80% of context window
    target_ratio=0.2,       # retain 20% after compaction
)

# Manual tool-based compaction (agent calls compact_conversation explicitly)
tool_summarization = create_summarization_tool_middleware(
    model="anthropic:claude-haiku-4-5-20251001",
)
```

`SummarizationToolMiddleware` exposes a `compact_conversation` tool the agent can call explicitly. `SummarizationMiddleware` triggers automatically.

### SkillsMiddleware

Loads SKILL.md files into the agent's system prompt at startup.

```python
from deepagents.middleware.skills import SkillsMiddleware

skills_mw = SkillsMiddleware()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    skills=["/skills/falkordb/", "/skills/deepagents/"],  # paths relative to backend root
    middleware=[skills_mw],
)
```

When using `FilesystemBackend`, skill files are loaded from disk. When using `StateBackend` (default), provide skill file contents via `invoke(files={...})`.

### MemoryMiddleware

Loads AGENTS.md files into the system prompt for long-term memory.

```python
agent = create_deep_agent(
    memory=["/memory/AGENTS.md"],
)
```

---

## 5. Backends

Backends control where file operations and execution happen.

### StateBackend (default)

Ephemeral, in-memory, thread-local. Resets between invocations unless a `checkpointer` is used.

```python
from deepagents.backends import StateBackend
backend = StateBackend()
```

### FilesystemBackend

Real filesystem access. Use `root_dir` to sandbox.

```python
from deepagents.backends import FilesystemBackend
backend = FilesystemBackend(root_dir="/home/user/myproject")
```

### StoreBackend

Persistent, cross-thread storage using LangGraph's `BaseStore`. Requires `store=` to be passed to `create_deep_agent`.

```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
backend = StoreBackend()

agent = create_deep_agent(
    backend=backend,
    store=store,
)
```

### CompositeBackend

Routes file operations to different backends based on path prefix.

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

backend = CompositeBackend(
    default=StateBackend(),           # ephemeral for most files
    routes={
        "/memories/": StoreBackend(), # persistent for /memories/ prefix
        "/cache/": StateBackend(),    # explicit ephemeral for /cache/
    }
)
```

### LocalShellBackend

Full filesystem + unrestricted shell execution. **Do not use in FOSRA** — use `FilesystemBackend` with `allow_write=False` instead.

---

## 6. FOSRA-Specific Agent Setup

```python
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.tools import tool

# --- Tool definitions ---

@tool
def run_retrieval_pipeline(query: str) -> str:
    """Search documentation and codebase for information relevant to the query.
    Use for any question about code, APIs, or documentation.
    Returns assembled context ready for answering the question.
    """
    result = fosra_retrieval_pipeline.invoke({"user_query": query})
    return result["context"]

@tool  
def check_coverage(query: str, context: str) -> str:
    """Check whether the retrieved context sufficiently answers the query.
    Returns a JSON object with coverage assessment and uncovered sub-queries.
    """
    return coverage_check_pipeline.invoke({"query": query, "context": context})

# --- Middleware ---

fs_middleware = FilesystemMiddleware(
    backend=FilesystemBackend(root_dir=PROJECT_ROOT),
    tool_token_limit_before_evict=20000,
)

summarization = create_summarization_middleware(
    token_threshold=0.8,   # compact at 80% context usage
    target_ratio=0.2,      # retain ~20% after compaction
)

# --- Agent ---

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",  # or local model via ollama
    tools=[run_retrieval_pipeline, check_coverage],
    system_prompt=AGENT_SYSTEM_PROMPT,
    middleware=[fs_middleware, summarization],
    subagents=None,         # flat tool list — no sub-agents
    checkpointer=checkpointer,   # for session persistence
    debug=False,
)
```

### Invoking with session persistence

```python
config = {"configurable": {"thread_id": session_id}}

# First turn
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": user_message}]},
    config=config,
)

# Subsequent turns — history is managed by checkpointer, just add new message
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": follow_up_message}]},
    config=config,
)
```

---

## 7. Subagents (Reference — not for FOSRA default)

Only use subagents with larger models (>= 30B). For 9B models, flat tool list is always better.

```python
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

research_agent = SubAgent(
    name="research_agent",
    description="Specialized agent for deep documentation research. Use when the query requires reading multiple documents.",
    prompt="You are a documentation research specialist. Search thoroughly and return a structured summary.",
    tools=[search_docs_tool],
    model="anthropic:claude-sonnet-4-20250514",
)

main_agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    subagents=[research_agent],
)
```

---

## 8. Human-in-the-Loop (interrupt_on)

Pause agent before specific tool calls for approval:

```python
agent = create_deep_agent(
    tools=[write_file_tool, run_retrieval_pipeline],
    interrupt_on={
        "write_file": True,     # always interrupt before writes
        "execute": True,        # always interrupt before shell execution
    }
)
```

---

## 9. Model String Formats

```python
# Anthropic
model = "anthropic:claude-sonnet-4-20250514"
model = "anthropic:claude-haiku-4-5-20251001"

# OpenAI
model = "openai:gpt-4o"

# Ollama (local)
model = "ollama:qwen2.5-coder:9b"

# Default (if None): claude-sonnet-4-5-20250929
```

---

## 10. Common Patterns and Pitfalls

### Pitfall: Subagents with small models

```python
# WRONG — 9B model loses context at each delegation hop
agent = create_deep_agent(
    model="ollama:qwen2.5-coder:9b",
    subagents=[research_agent, code_agent],  # ❌
)

# CORRECT — flat tools, single agent
agent = create_deep_agent(
    model="ollama:qwen2.5-coder:9b",
    tools=[research_tool, code_tool],  # ✓
)
```

### Pitfall: Not using checkpointer for sessions

```python
# WRONG — history lost between calls
result1 = agent.invoke({"messages": [...]})
result2 = agent.invoke({"messages": [...]})  # forgets result1

# CORRECT — use thread_id + checkpointer
from langgraph.checkpoint.memory import MemorySaver
agent = create_deep_agent(..., checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-123"}}
result1 = agent.invoke({"messages": [...]}, config=config)
result2 = agent.invoke({"messages": [...]}, config=config)  # remembers
```

### Pitfall: Tool docstrings

The agent decides which tool to call entirely from the docstring. Write docstrings as instructions to the model, not descriptions for humans.

```python
# WRONG — vague
@tool
def search(q: str) -> str:
    """Search the knowledge base."""
    ...

# CORRECT — specific, action-oriented
@tool
def run_retrieval_pipeline(query: str) -> str:
    """Search the project's documentation and source code for context relevant to the query.
    Use this tool for ANY question about how code works, API usage, configuration,
    or implementation details. Returns assembled context from docs + code graph.
    
    Args:
        query: Natural language question. Be specific — include function names,
               class names, or file paths if known.
    """
    ...
```

### Pattern: Tool result eviction awareness

When `tool_token_limit_before_evict` is exceeded (default 20,000 tokens), the middleware writes the result to a file and replaces it in context with a reference. The agent will then `read_file` to access the full content. This is automatic — no code change needed. Set `tool_token_limit_before_evict=None` to disable.

---

## 11. Key Imports

```python
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.subagents import SubAgent, SubAgentMiddleware
from deepagents.backends import (
    StateBackend,
    StoreBackend,
    FilesystemBackend,
    CompositeBackend,
)
from deepagents.backends.protocol import BackendProtocol
```
