from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from falkordb import FalkorDB
from loguru import logger

from backend.src.services.session.tools import (
    RetrievalResultStore,
    create_graph_tool,
    create_retrieval_tool,
)
from backend.src.services.session.utils.llm_utils import build_llm
from backend.src.settings import LLMConfig, settings
from backend.src.settings.config import EmbedderConfig, VectorStoreConfig
from backend.src.settings.fosra_paths import fosra_paths

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from backend.src.settings.config import UserPreferences


def create_fosra_agent(
    user_prefs: UserPreferences,
    system_prompt: str | None = None,
    backend: Any | None = None,
    checkpointer: Any | None = None,
    llm_config: LLMConfig | None = None,
) -> tuple[CompiledStateGraph[Any, Any, Any, Any], RetrievalResultStore]:
    if system_prompt is None:
        from backend.src.services.session.utils.prompts import (
            FOSRA_AGENT_SYSTEM_PROMPT,
        )

        system_prompt = FOSRA_AGENT_SYSTEM_PROMPT

    if llm_config is None:
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
                model=settings.agent.fallback_model,
                api_key="not-needed",
                api_base=settings.agent.fallback_api_base,
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

    graph_tool = create_graph_tool(
        FalkorDB(host=settings.falkordb.host, port=settings.falkordb.port),
        user_prefs.embedder or EmbedderConfig(),
    )

    logger.info(
        "Creating FOSRA agent with model={}/{} backend={}",
        llm_config.provider,
        llm_config.model,
        type(backend).__name__ if backend else "none",
    )

    mw_backend = FilesystemBackend(root_dir=fosra_paths.data_dir)

    research_subagent = dict(
        name="Research",
        description="Performs deep research on a topic using web search and fetch tools. Use for fact-checking, background research, and information gathering.",
        system_prompt="You are a research assistant. Your role is to gather comprehensive, accurate information on a given topic using available tools. Be thorough and cite sources where possible.",
        tools=[],
        model=f"{llm_config.provider}:{llm_config.model}",
    )

    code_analysis_subagent = dict(
        name="Code Analysis",
        description="Analyzes code structure, call chains, and relationships. Use for understanding codebases, finding functions, tracing dependencies, and refactoring planning.",
        system_prompt="You are a code analysis specialist. Your role is to analyze code structure, find functions, trace call chains, and help understand codebases. Use code graph tools to explore.",
        tools=[graph_tool],
        model=f"{llm_config.provider}:{llm_config.model}",
    )

    kwargs: dict[str, Any] = {
        "model": llm,
        "tools": [retrieval_tool],
        "system_prompt": system_prompt,
        "backend": backend or mw_backend,
        "memory": [
            str(Path(__file__).parent / "fixtures" / "AGENTS.md"),
        ],
        "skills": [str(fosra_paths.skills_dir)],
        "subagents": [research_subagent, code_analysis_subagent],
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    agent = create_deep_agent(**kwargs)

    return agent, result_store
