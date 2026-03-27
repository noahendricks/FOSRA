"""
provider registry — transforms models.dev data into tui-compatible Provider/Model shapes.

fetches from https://models.dev/api.json on startup and caches.
priority providers (minimax-coding-plan, ollama-cloud) are hardcoded as fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from backend.src.api.schemas.tui_schemas import (
    Model,
    ModelApi,
    ModelCapabilities,
    ModelCapabilitiesInput,
    ModelCapabilitiesOutput,
    ModelCapabilitiesInterleaved,
    ModelCost,
    ModelCostCache,
    ModelLimit,
    Provider,
)

logger = logging.getLogger(__name__)


# PROVIDER ORDERING
# first provider is the default when no selection is persisted
PRIORITY_PROVIDERS = ["minimax-coding-plan", "ollama-cloud"]


def _modalities_to_bools(
    modalities: dict[str, list[str]] | None, direction: str
) -> dict[str, bool]:
    items = modalities.get(direction, []) if modalities else []
    return {
        "text": "text" in items,
        "audio": "audio" in items,
        "image": "image" in items,
        "video": "video" in items,
        "pdf": "pdf" in items,
    }


def _transform_model(
    model_data: dict[str, Any],
    provider_id: str,
    api_id: str,
    api_url: str,
    api_npm: str,
) -> Model:
    """convert a models.dev model entry into a tui Model."""
    cost = model_data.get("cost", {})
    modalities = model_data.get("modalities")
    inp = _modalities_to_bools(modalities, "input")
    out = _modalities_to_bools(modalities, "output")

    interleaved_raw = model_data.get("interleaved", False)
    if isinstance(interleaved_raw, dict):
        interleaved: bool | ModelCapabilitiesInterleaved = ModelCapabilitiesInterleaved(
            field=interleaved_raw["field"]
        )
    else:
        interleaved = bool(interleaved_raw)

    status_raw = model_data.get("status")
    status = status_raw if status_raw in ("alpha", "beta", "deprecated") else "active"

    return Model(
        id=model_data["id"],
        providerID=provider_id,
        api=ModelApi(id=api_id, url=api_url, npm=api_npm),
        name=model_data.get("name", model_data["id"]),
        family=model_data.get("family"),
        capabilities=ModelCapabilities(
            temperature=model_data.get("temperature", True),
            reasoning=model_data.get("reasoning", False),
            attachment=model_data.get("attachment", False),
            toolcall=model_data.get("tool_call", False),
            input=ModelCapabilitiesInput(**inp),
            output=ModelCapabilitiesOutput(**out),
            interleaved=interleaved,
        ),
        cost=ModelCost(
            input=cost.get("input", 0),
            output=cost.get("output", 0),
            cache=ModelCostCache(
                read=cost.get("cache_read", 0),
                write=cost.get("cache_write", 0),
            ),
        ),
        limit=ModelLimit(
            context=model_data.get("limit", {}).get("context", 128000),
            input=model_data.get("limit", {}).get("input"),
            output=model_data.get("limit", {}).get("output", 4096),
        ),
        status=status,
        options=model_data.get("options", {}),
        headers=model_data.get("headers", {}),
        releaseDate=model_data.get("release_date", "2025-01-01"),
    )


def _transform_provider(
    provider_data: dict[str, Any],
) -> Provider:
    """convert a models.dev provider entry into a tui Provider."""
    pid = provider_data["id"]
    api_url = provider_data.get("api", "")
    api_npm = provider_data.get("npm", "")
    models_raw = provider_data.get("models", {})

    models = {}
    for model_id, model_data in models_raw.items():
        models[model_id] = _transform_model(
            model_data,
            provider_id=pid,
            api_id=pid,
            api_url=api_url,
            api_npm=api_npm,
        )

    return Provider(
        id=pid,
        name=provider_data.get("name", pid),
        source="env",
        env=provider_data.get("env", []),
        options={},
        models=models,
    )


# =========================================================================
# PROVIDER DATA
# =========================================================================

_PROVIDER_DATA: dict[str, dict[str, Any]] = {
    "minimax-coding-plan": {
        "id": "minimax-coding-plan",
        "env": ["MINIMAX_API_KEY"],
        "npm": "@ai-sdk/anthropic",
        "api": "https://api.minimax.io/anthropic/v1",
        "name": "MiniMax Coding Plan (minimax.io)",
        "models": {
            "MiniMax-M2.7": {
                "id": "MiniMax-M2.7",
                "name": "MiniMax-M2.7",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2026-03-18",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "limit": {"context": 204800, "output": 131072},
            },
            "MiniMax-M2.5": {
                "id": "MiniMax-M2.5",
                "name": "MiniMax-M2.5",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2026-02-12",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "limit": {"context": 204800, "output": 131072},
            },
            "MiniMax-M2.7-highspeed": {
                "id": "MiniMax-M2.7-highspeed",
                "name": "MiniMax-M2.7-highspeed",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2026-03-18",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "limit": {"context": 204800, "output": 131072},
            },
            "MiniMax-M2.5-highspeed": {
                "id": "MiniMax-M2.5-highspeed",
                "name": "MiniMax-M2.5-highspeed",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2026-02-13",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "limit": {"context": 204800, "output": 131072},
            },
            "MiniMax-M2.1": {
                "id": "MiniMax-M2.1",
                "name": "MiniMax-M2.1",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2025-12-23",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0},
                "limit": {"context": 204800, "output": 131072},
            },
            "MiniMax-M2": {
                "id": "MiniMax-M2",
                "name": "MiniMax-M2",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "temperature": True,
                "release_date": "2025-10-27",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0},
                "limit": {"context": 196608, "output": 128000},
            },
        },
    },
    "ollama-cloud": {
        "id": "ollama-cloud",
        "env": ["OLLAMA_API_KEY"],
        "npm": "@ai-sdk/openai-compatible",
        "api": "https://ollama.com/v1",
        "name": "Ollama Cloud",
        "models": {
            "minimax-m2.7": {
                "id": "minimax-m2.7",
                "name": "minimax-m2.7",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2026-03-18",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 204800, "output": 131072},
            },
            "minimax-m2.5": {
                "id": "minimax-m2.5",
                "name": "minimax-m2.5",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2026-02-12",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 204800, "output": 131072},
            },
            "minimax-m2.1": {
                "id": "minimax-m2.1",
                "name": "minimax-m2.1",
                "family": "minimax",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-12-23",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 204800, "output": 131072},
            },
            "deepseek-v3.2": {
                "id": "deepseek-v3.2",
                "name": "deepseek-v3.2",
                "family": "deepseek",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-06-15",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 163840, "output": 65536},
            },
            "deepseek-v3.1:671b": {
                "id": "deepseek-v3.1:671b",
                "name": "deepseek-v3.1:671b",
                "family": "deepseek",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-08-21",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 163840, "output": 163840},
            },
            "qwen3-coder:480b": {
                "id": "qwen3-coder:480b",
                "name": "qwen3-coder:480b",
                "family": "qwen",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2025-07-22",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 65536},
            },
            "qwen3-coder-next": {
                "id": "qwen3-coder-next",
                "name": "qwen3-coder-next",
                "family": "qwen",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2026-02-02",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 65536},
            },
            "kimi-k2.5": {
                "id": "kimi-k2.5",
                "name": "kimi-k2.5",
                "family": "kimi",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2026-01-27",
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "kimi-k2:1t": {
                "id": "kimi-k2:1t",
                "name": "kimi-k2:1t",
                "family": "kimi",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2025-07-11",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "kimi-k2-thinking": {
                "id": "kimi-k2-thinking",
                "name": "kimi-k2-thinking",
                "family": "kimi-thinking",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-11-06",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "glm-5": {
                "id": "glm-5",
                "name": "glm-5",
                "family": "glm",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "interleaved": {"field": "reasoning_content"},
                "release_date": "2026-02-11",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 202752, "output": 131072},
            },
            "glm-4.7": {
                "id": "glm-4.7",
                "name": "glm-4.7",
                "family": "glm",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-12-22",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 202752, "output": 131072},
            },
            "glm-4.6": {
                "id": "glm-4.6",
                "name": "glm-4.6",
                "family": "glm",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-09-29",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 202752, "output": 131072},
            },
            "nemotron-3-super": {
                "id": "nemotron-3-super",
                "name": "nemotron-3-super",
                "family": "nemotron",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2026-03-11",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 65536},
            },
            "nemotron-3-nano:30b": {
                "id": "nemotron-3-nano:30b",
                "name": "nemotron-3-nano:30b",
                "family": "nemotron",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-12-15",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 1048576, "output": 131072},
            },
            "devstral-2:123b": {
                "id": "devstral-2:123b",
                "name": "devstral-2:123b",
                "family": "devstral",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2025-12-09",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "devstral-small-2:24b": {
                "id": "devstral-small-2:24b",
                "name": "devstral-small-2:24b",
                "family": "devstral",
                "attachment": True,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2025-12-09",
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "mistral-large-3:675b": {
                "id": "mistral-large-3:675b",
                "name": "mistral-large-3:675b",
                "family": "mistral-large",
                "attachment": True,
                "reasoning": False,
                "tool_call": True,
                "release_date": "2025-12-02",
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 262144, "output": 262144},
            },
            "cogito-2.1:671b": {
                "id": "cogito-2.1:671b",
                "name": "cogito-2.1:671b",
                "family": "cogito",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-11-19",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 163840, "output": 32000},
            },
            "qwen3.5:397b": {
                "id": "qwen3.5:397b",
                "name": "qwen3.5:397b",
                "family": "qwen",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "interleaved": {"field": "reasoning_details"},
                "release_date": "2026-02-15",
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 262144, "output": 81920},
            },
            "qwen3-next:80b": {
                "id": "qwen3-next:80b",
                "name": "qwen3-next:80b",
                "family": "qwen",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-09-15",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 32768},
            },
            "gemini-3-flash-preview": {
                "id": "gemini-3-flash-preview",
                "name": "gemini-3-flash-preview",
                "family": "gemini-flash",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-12-17",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 1048576, "output": 65536},
            },
            "gpt-oss:120b": {
                "id": "gpt-oss:120b",
                "name": "gpt-oss:120b",
                "family": "gpt-oss",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-08-05",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 32768},
            },
            "gpt-oss:20b": {
                "id": "gpt-oss:20b",
                "name": "gpt-oss:20b",
                "family": "gpt-oss",
                "attachment": False,
                "reasoning": True,
                "tool_call": True,
                "release_date": "2025-08-05",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 32768},
            },
        },
    },
}

# =========================================================================
# PUBLIC API
# =========================================================================

_providers_cache: list[Provider] | None = None


def _build_providers() -> list[Provider]:
    global _providers_cache
    if _providers_cache is not None:
        return _providers_cache

    providers = []
    for pid in PRIORITY_PROVIDERS:
        raw = _PROVIDER_DATA.get(pid)
        if raw:
            providers.append(_transform_provider(raw))

    _providers_cache = providers
    return providers


def get_all_providers() -> list[dict[str, Any]]:
    """all providers as dicts, ordered by priority."""
    return [p.model_dump() for p in _build_providers()]


def get_provider_defaults() -> dict[str, str]:
    """providerID → default modelID mapping."""
    defaults = {}
    for p in _build_providers():
        if p.models:
            defaults[p.id] = next(iter(p.models))
    return defaults


def get_connected_providers() -> list[str]:
    """providers whose env var is set (or all if running as backend proxy)."""
    connected = []
    for p in _build_providers():
        # if any required env var is set, mark as connected
        if not p.env or any(os.environ.get(var) for var in p.env):
            connected.append(p.id)
    return connected


def get_config_providers_response() -> dict[str, Any]:
    """/config/providers response."""
    return {
        "providers": get_all_providers(),
        "default": get_provider_defaults(),
    }


def get_provider_list_response() -> dict[str, Any]:
    """/provider response."""
    return {
        "all": get_all_providers(),
        "default": get_provider_defaults(),
        "connected": get_connected_providers(),
    }
