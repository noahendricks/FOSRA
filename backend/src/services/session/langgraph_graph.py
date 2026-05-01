"""LangGraph graph factory for LangGraph Studio / Agent Chat UI.

This module provides the entry point that langgraph dev expects. The exported
`graph` is synchronous — langgraph dev calls it directly without awaiting.

Debug features:
- Checkpointing via PostgresSaver for time-travel debugging
- Interrupt breakpoints on file editing/execution tools
- State inspection and forking via get_state/update_state
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

from langgraph.graph.state import Checkpointer, CompiledStateGraph, RunnableConfig

if TYPE_CHECKING:
    pass


_agent_cache: CompiledStateGraph | None = None
_cache_lock = threading.Lock()


def _get_checkpointer() -> Checkpointer | None:
    """Get the Postgres checkpointer from global infra or create InMemorySaver."""
    try:
        from backend.src.api.lifecycle import global_infra

        if global_infra.checkpointer is not None:
            return global_infra.checkpointer
    except Exception:
        pass

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:
        return None


def _get_interrupt_config() -> dict[str, bool | dict[str, Any]]:
    env_interrupt = os.environ.get("FOSRA_INTERRUPT_ON", "")
    if env_interrupt:
        return {tool: True for tool in env_interrupt.split(",")}

    return {
        "edit_file": True,
        "write_file": True,
        "execute": True,
        "mcp__tool_call": True,
    }


def graph(config: RunnableConfig | None = None) -> CompiledStateGraph:
    """Synchronous entry point for langgraph dev.

    langgraph dev imports and calls this directly. We build the agent on first
    call using a thread pool (since agent creation is async) and cache the result.
    """
    global _agent_cache

    if _agent_cache is not None:
        return _agent_cache

    with _cache_lock:
        if _agent_cache is not None:
            return _agent_cache

        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def _build():
            return asyncio.run(_build_agent())

        with ThreadPoolExecutor(max_workers=1) as executor:
            _agent_cache = executor.submit(_build).result(timeout=120)

        return _agent_cache


async def _build_agent() -> CompiledStateGraph:
    """Build and return the compiled fosra agent graph."""
    from backend.src.domain.schemas import LLMConfig, UserPreferences
    from backend.src.services.session.agent_service import create_fosra_agent

    llm_config = LLMConfig(
        provider="MINIMAX",
        model="MiniMax-M2.5",
        api_key="sk-cp-DWNxzHNSG1EZbRkGwYJrtMTIkxX4zBq_9aCySisXhO97NPBey0AxZY_YL9ctCXBCPqFRed5KE7HLR55T_lav4TG6_uNI_HTOyI6sYxXj-gRzrXaRX9rdCaI",
        api_base="https://api.minimax.io/v1",
    )
    user_prefs = UserPreferences(llm_default=llm_config)

    checkpointer = _get_checkpointer()
    interrupt_on = _get_interrupt_config()

    agent, _ = await create_fosra_agent(
        user_prefs=user_prefs,
        enable_ingest_tools=True,
        checkpointer=checkpointer,
        interrupt_on=interrupt_on,
    )
    return agent
