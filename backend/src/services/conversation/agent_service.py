"""FOSRA agent creation using DeepAgents.

Creates a DeepAgent wrapping the user's configured LLM with the
``search_knowledge_base`` retrieval tool.  Built-in deepagents
middleware (FilesystemMiddleware, TodoListMiddleware, Summarization)
is always applied automatically.

Usage::

    agent, result_store = create_fosra_agent(user_prefs)
    async for msg, meta in agent.astream(
        {"messages": lc_messages},
        stream_mode="messages",
    ):
        ...
    # After streaming, result_store.chunks has the retrieved chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from loguru import logger

from backend.src.services.conversation.tools import (
    RetrievalResultStore,
    create_retrieval_tool,
)
from backend.src.services.conversation.utils.llm_utils import build_llm
from backend.src.settings import LLMConfig
from backend.src.settings.config import EmbedderConfig, VectorStoreConfig

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from backend.src.settings.config import UserPreferences


def create_fosra_agent(
    user_prefs: UserPreferences,
    system_prompt: str | None = None,
    backend: Any | None = None,
) -> tuple[CompiledStateGraph, RetrievalResultStore]:
    """Create a FOSRA agent with retrieval capabilities.

    Parameters
    ----------
    user_prefs:
        The current user's preferences (LLM configs, embedder, vector
        store, reranker settings).
    system_prompt:
        Custom system prompt.  If ``None``, uses
        ``FOSRA_AGENT_SYSTEM_PROMPT`` from prompts module.
    backend:
        Optional filesystem backend for coding mode. When provided,
        the agent gets read/write/edit/grep/glob tools.

    Returns
    -------
    tuple[CompiledStateGraph, RetrievalResultStore]
        The compiled agent and a mutable store that the retrieval tool
        populates with chunks.  The caller can read
        ``result_store.chunks`` after the agent finishes to build
        source-group SSE events.
    """

    # -- Resolve prompt ------------------------------------------------
    if system_prompt is None:
        from backend.src.services.conversation.utils.prompts import (
            FOSRA_AGENT_SYSTEM_PROMPT,
        )

        system_prompt = FOSRA_AGENT_SYSTEM_PROMPT

    # -- Resolve LLM ---------------------------------------------------
    llm_config: LLMConfig
    if user_prefs:
        for cfg in (
            user_prefs.llm_default,
            user_prefs.llm_logic,
            user_prefs.llm_fast,
            user_prefs.llm_heavy,
        ):
            if cfg is not None:
                llm_config = cfg
                break
        else:
            llm_config = LLMConfig(
                provider="openai",
                model="Qwen3.5-35B-A3B-Q4_K_M.gguf",
                api_key="not-needed",
                api_base="http://localhost:8045/v1",
            )
    else:
        llm_config = LLMConfig(
            provider="openai",
            model="Qwen3.5-35B-A3B-Q4_K_M.gguf",
            api_key="not-needed",
            api_base="http://localhost:8045/v1",
        )
    llm = build_llm(llm_config)

    # -- Build retrieval tool ------------------------------------------
    result_store = RetrievalResultStore()

    retrieval_tool = create_retrieval_tool(
        llm_config=llm_config,
        embedder_config=user_prefs.embedder or EmbedderConfig(),
        vector_config=user_prefs.vector_store or VectorStoreConfig(),
        reranker_config=user_prefs.reranker,
        token_budget=(user_prefs.chunker.token_budget if user_prefs.chunker else 4096),
        max_iterations=3,
        result_store=result_store,
    )

    # -- Create agent --------------------------------------------------
    #
    # We pass the ChatLiteLLM instance directly as the model.
    # deepagents accepts any BaseChatModel and will bind tools to it.
    #
    # Built-in middleware (TodoList, Filesystem, Summarization, etc.)
    # is applied automatically — we get read_file, ls, glob, grep
    # for free.  The only custom tool we add is retrieval.
    #
    logger.info(
        "Creating FOSRA agent with model={}/{} backend={}",
        llm_config.provider,
        llm_config.model,
        type(backend).__name__ if backend else "none",
    )

    kwargs: dict[str, Any] = {
        "model": llm,
        "tools": [retrieval_tool],
        "system_prompt": system_prompt,
    }
    if backend is not None:
        kwargs["backend"] = backend

    agent = create_deep_agent(**kwargs)

    return agent, result_store
