from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from falkordb import FalkorDB
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger

from backend.src.services.session.langgraph_graph import _get_checkpointer
from backend.src.services.session.tools import (
    RetrievalResultStore,
    create_graph_tool,
    create_retrieval_tool,
)
from backend.src.services.session.utils.llm_utils import build_llm
from backend.src.settings import LLMConfig, settings
from backend.src.settings.config import EmbedderConfig, VectorStoreConfig
from backend.src.settings.fosra_paths import fosra_paths

# from langchain.agents.middleware import

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from backend.src.settings.config import UserPreferences


async def create_fosra_agent(
    user_prefs: UserPreferences,
    system_prompt: str | None = None,
    backend: Any | None = None,
    checkpointer: Any | None = None,
    llm_config: LLMConfig | None = None,
    enable_ingest_tools: bool = True,
    session_factory: Any | None = None,
    falkordb_client: Any | None = None,
    interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
) -> tuple[CompiledStateGraph[Any, Any, Any, Any], RetrievalResultStore]:
    if system_prompt is None:
        from backend.src.services.session.utils.prompts import (
            FOSRA_AGENT_SYSTEM_PROMPT,
        )

        system_prompt = FOSRA_AGENT_SYSTEM_PROMPT

    ...
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
    ...

    llm = build_llm(llm_config)

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

    ...

    graph_tool = create_graph_tool(
        FalkorDB(host=settings.falkordb.host, port=settings.falkordb.port),
        user_prefs.embedder or EmbedderConfig(),
    )

    all_tools: list[Any] = [retrieval_tool]

    ...

    # allow direct injection of infra deps (for debug/testing) or fall back to global_infra
    if enable_ingest_tools:
        ...
        from backend.src.api.lifecycle import global_infra
        from backend.src.services.session.tools import (
            create_ingest_codebase_tool,
            create_ingest_file_tool,
        )
        from backend.src.settings import ChunkerConfig

        ...

        sf = session_factory or global_infra.session_factory

        fb = falkordb_client or global_infra.falkordb_client

        ...

        print(f"[sf] exists: {sf}")
        print(f"[fb] exists: {fb}")

        if sf is not None and fb is not None:
            embed_cfg = user_prefs.embedder or EmbedderConfig()
            chunker_cfg = user_prefs.chunker if user_prefs.chunker else ChunkerConfig()
            vector_cfg = (
                user_prefs.vector_store
                if user_prefs.vector_store
                else VectorStoreConfig()
            )

            ingest_codebase_tool = create_ingest_codebase_tool(
                session_factory=sf,
                falkordb_client=fb,
                embedder_config=embed_cfg,
            )

            ingest_file_tool = create_ingest_file_tool(
                session_factory=sf,
                embedder_config=embed_cfg,
                vector_config=vector_cfg,
                chunker_config=chunker_cfg,
            )

            all_tools.extend([ingest_codebase_tool, ingest_file_tool])
            logger.info("Ingestion tools enabled in agent")
        else:
            logger.warning("Infra not available; ingestion tools disabled")

    logger.info(
        "Creating FOSRA agent with model={}/{} backend={} tools={}",
        llm_config.provider,
        llm_config.model,
        type(backend).__name__ if backend else "none",
        len(all_tools),
    )

    mw_backend = FilesystemBackend(root_dir=fosra_paths.data_dir)

    # Build the litellm model string for subagents
    from backend.src.services.session.utils.llm_utils import _build_model_string

    subagent_model_string = _build_model_string(llm_config.provider, llm_config.model)

    # Create a ChatLiteLLM instance for subagents with the correct api_base
    from langchain_community.chat_models.litellm import ChatLiteLLM

    # For MiniMax, use custom_llm_provider to ensure litellm routes correctly
    extra_kwargs = {}
    if llm_config.provider.upper() in ("MINIMAX", "MINIMAX-CODING-PLAN"):
        extra_kwargs["custom_llm_provider"] = "openai"

    subagent_llm = ChatLiteLLM(
        model=subagent_model_string,
        api_key=llm_config.get_api_key_value(),
        api_base=llm_config.api_base,
        **extra_kwargs,
    )

    research_subagent = dict(
        name="Research",
        description="Performs deep research on a topic using web search and fetch tools. Use for fact-checking, background research, and information gathering.",
        system_prompt="You are a research assistant. Your role is to gather comprehensive, accurate information on a given topic using available tools. Be thorough and cite sources where possible.",
        tools=[],
        model=subagent_llm,  # Use ChatLiteLLM instance instead of string
    )

    code_analysis_subagent = dict(
        name="Code Analysis",
        description="Analyzes code structure, call chains, and relationships. Use for understanding codebases, finding functions, tracing dependencies, and refactoring planning.",
        system_prompt="You are a code analysis specialist. Your role is to analyze code structure, find functions, trace call chains, and help understand codebases. Use code graph tools to explore.",
        tools=[graph_tool],
        model=subagent_llm,  # Use ChatLiteLLM instance instead of string
    )

    kwargs: dict[str, Any] = {
        "model": llm,
        "tools": all_tools,
        "system_prompt": system_prompt,
        "backend": backend or mw_backend,
        "memory": [
            str(Path(__file__).parent / "fixtures" / "AGENTS.md"),
        ],
        "skills": [str(fosra_paths.skills_dir)],
        "subagents": [research_subagent, code_analysis_subagent],
    }

    checkpointer = _get_checkpointer()

    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    else:
        print("[ENTERED FALLBACK]")
        conn_string = "host=localhost port=5432 dbname=postgres user=postgres"
        saver_ctx = AsyncPostgresSaver.from_conn_string(conn_string)
        async with saver_ctx as saver:
            await saver.setup()
            kwargs["checkpointer"] = saver

    if interrupt_on is not None:
        kwargs["interrupt_on"] = interrupt_on

    print("checkpoint exists:")
    print(kwargs["checkpointer"] is not None)

    agent = create_deep_agent(
        debug=False,
        **kwargs,
    )

    return agent, result_store
