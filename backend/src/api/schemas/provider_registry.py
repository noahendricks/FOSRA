"""
provider registry — transforms models.dev data into tui-compatible Provider/Model shapes.

fetches from https://models.dev/api.json on startup and caches.
priority providers (minimax-coding-plan, ollama-cloud) are hardcoded as fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging

from loguru import logger as _loguru
import os
from pathlib import Path
from typing import Any

import httpx

from backend.src.api.schemas.tui_schemas import (
    Model,
    ModelApi,
    ModelCapabilities,
    ModelCapabilitiesInput,
    ModelCapabilitiesInterleaved,
    ModelCapabilitiesOutput,
    ModelCost,
    ModelCostCache,
    ModelLimit,
    Provider,
)

# Re-export loguru logger so existing logger.info/warning/error calls work
logger = _loguru


# =========================================================================
# MODELS.DEV CONFIGURATION
# =========================================================================

MODELS_DEV_URL = os.environ.get("MODELS_DEV_URL", "https://models.dev/api.json")
MODELS_CACHE_DIR = Path.home() / ".cache" / "fosra"
MODELS_CACHE_PATH = MODELS_CACHE_DIR / "models.json"
MODELS_FETCH_TIMEOUT = 10.0
MODELS_REFRESH_INTERVAL = 60 * 60  # 1 hour


# =========================================================================
# MODELS.DEV CACHE
# =========================================================================


def _ensure_cache_dir() -> None:
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_models_cache() -> dict[str, Any] | None:
    """Read cached models.dev data if available."""
    if not MODELS_CACHE_PATH.exists():
        return None
    try:
        with open(MODELS_CACHE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read models cache: {}", e)
        return None


def _write_models_cache(data: dict[str, Any]) -> None:
    """Write models.dev data to cache."""
    try:
        _ensure_cache_dir()
        with open(MODELS_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Failed to write models cache: {}", e)


async def _fetch_models_from_api() -> dict[str, Any] | None:
    """Fetch models.dev data from the API endpoint."""
    try:
        async with httpx.AsyncClient(timeout=MODELS_FETCH_TIMEOUT) as client:
            response = await client.get(
                MODELS_DEV_URL,
                headers={"User-Agent": "fosra-backend/1.0"},
            )
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "models.dev API returned status {}: {}",
                response.status_code,
                response.text[:200],
            )
    except Exception as e:
        logger.warning("Failed to fetch models.dev: {}", e)
    return None


async def _get_models_dev_data() -> dict[str, Any]:
    """
    Get models.dev data: cache first, then fallback to API.

    Priority providers from _PROVIDER_DATA are merged on top of fetched data.
    """
    # try cache first
    cached = _read_models_cache()
    if cached:
        logger.info("Using cached models.dev data")
        return cached

    # try API
    logger.info("Fetching models.dev data from API...")
    data = await _fetch_models_from_api()
    if data:
        _write_models_cache(data)
        return data

    # fallback to hardcoded data
    logger.warning("Using hardcoded fallback provider data")
    return {}


async def refresh_models_cache() -> bool:
    """Force refresh models.dev data from API. Returns True on success."""
    logger.info("Refreshing models.dev cache...")
    data = await _fetch_models_from_api()
    if data:
        _write_models_cache(data)
        # invalidate the providers cache so it rebuilds with new data
        global _providers_cache
        _providers_cache = None
        logger.info("models.dev cache refreshed successfully")
        return True
    logger.warning("models.dev cache refresh failed")
    return False


_refresh_task: asyncio.Task | None = None


def _start_models_refresh_task() -> None:
    """Start background task to periodically refresh models.dev data."""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return

    async def _periodic_refresh():
        while True:
            await asyncio.sleep(MODELS_REFRESH_INTERVAL)
            await refresh_models_cache()

    _refresh_task = asyncio.create_task(_periodic_refresh())
    logger.info("Started models.dev periodic refresh task")


# first provider is the default when no selection is persisted
PRIORITY_PROVIDERS = ["ollama", "local", "ollama-cloud"]


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
    "local": {
        "id": "local",
        "env": [],
        "npm": "@ai-sdk/openai-compatible",
        "api": "http://localhost:8045/v1",
        "name": "Local (localhost:8045)",
        "models": {
            "local-model": {
                "id": "local-model",
                "name": "Local Model",
                "family": "local",
                "attachment": False,
                "reasoning": False,
                "tool_call": True,
                "temperature": True,
                "release_date": "2025-01-01",
                "modalities": {"input": ["text"], "output": ["text"]},
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "limit": {"context": 128000, "output": 4096},
            },
        },
    },
    "minimax-coding-plan": {
        "id": "minimax-coding-plan",
        "env": ["MINIMAX_API_KEY"],
        "npm": "@ai-sdk/anthropic",
        "api": "https://api.minimax.io/v1",
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
# MODELS.DEV DATA (sync cache for runtime use)
# =========================================================================

_models_dev_data: dict[str, Any] = {}


def _load_models_dev_data() -> dict[str, Any]:
    """
    Load models.dev data: cache → _PROVIDER_DATA fallback.
    Called synchronously at module load time.
    """
    cached = _read_models_cache()
    if cached:
        return cached
    # use hardcoded fallback
    return {}


async def _init_models_dev() -> None:
    """
    Initialize models.dev data. Called on app startup.
    1. Try cache first (sync)
    2. If no cache, try API fetch
    3. Start background refresh task
    """
    global _models_dev_data, _providers_cache

    # load cache synchronously first
    _models_dev_data = _load_models_dev_data()

    # invalidate providers cache since data may have changed
    _providers_cache = None

    # if no cache, try to fetch from API
    if not _models_dev_data:
        logger.info("No cache found, fetching models.dev from API...")
        data = await _fetch_models_from_api()
        if data:
            _models_dev_data = data
            _write_models_cache(data)
            _providers_cache = None

    # start background refresh task
    _start_models_refresh_task()

    # initialize local Ollama provider
    await _init_local_ollama()

    logger.info("models.dev initialization complete")


def _build_providers_from_data(data: dict[str, Any]) -> list[Provider]:
    """Build Provider list from raw models.dev data, merging priority providers on top."""
    providers = []

    # first, add all providers from fetched data
    for pid, provider_data in data.items():
        if provider_data.get("models"):
            providers.append(_transform_provider(provider_data))

    # then, merge/override with priority providers from _PROVIDER_DATA
    for pid in PRIORITY_PROVIDERS:
        override = _PROVIDER_DATA.get(pid)
        if override:
            # find if this provider already exists from fetched data
            existing = next((p for p in providers if p.id == pid), None)
            if existing:
                # merge: override existing provider's data
                existing_dict = existing.model_dump()
                existing_dict.update(override)
                providers = [p for p in providers if p.id != pid]
                providers.append(_transform_provider(existing_dict))
            else:
                # add priority provider that wasn't in fetched data
                providers.append(_transform_provider(override))

    return providers


# =========================================================================
# LOCAL OLLAMA PROVIDER
# =========================================================================

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL_FETCH_TIMEOUT = 10.0

_ollama_provider: Provider | None = None


async def _fetch_local_ollama_models() -> tuple[list[dict[str, Any]], list[str]] | None:
    """Fetch available models from local Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_MODEL_FETCH_TIMEOUT) as client:
            response = await client.get(f"{OLLAMA_API_URL}/v1/models")
            if response.status_code != 200:
                return None
            data = response.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            if not model_ids:
                return None

            show_tasks = [
                client.post(
                    f"{OLLAMA_API_URL}/api/show",
                    json={"name": mid},
                    follow_redirects=True,
                )
                for mid in model_ids
            ]
            show_results = await asyncio.gather(*show_tasks)

            models = []
            for m in show_results:
                import json

                models.append(json.loads(m.content))

            return (models, model_ids)
    except Exception as e:
        logger.opt(exception=True).warning("Failed to fetch local Ollama models")
        return None
    return None


def _serialize_local_ollama(
    models: list[dict[str, Any]], model_ids: list[str]
) -> dict[str, Model]:
    """Serialize Ollama model data into Model dict, filtering embedding models."""
    serialized = {}

    for m, mid in zip(models, model_ids):
        c = m.get("capabilities", [])
        if "embedding" in c or "completion" not in c:
            continue

        family = m.get("details", {}).get("family")
        if family is None:
            family = m.get("model_info", {}).get("general.architecture")

        model = Model(
            id=mid,
            providerID="ollama",
            api=ModelApi(id="ollama", url=OLLAMA_API_URL, npm=""),
            name=m.get("model_info", {}).get("general.basename", mid.split(":")[0]),
            family=family,
            capabilities=ModelCapabilities(
                temperature="completion" in c,
                reasoning="thinking" in c,
                attachment=False,
                toolcall="tools" in c,
                input=ModelCapabilitiesInput(
                    text=True, audio=False, image=False, video=False, pdf=False
                ),
                output=ModelCapabilitiesOutput(
                    text=True, audio=False, image=False, video=False, pdf=False
                ),
                interleaved=False,
            ),
            cost=ModelCost(
                input=0,
                output=0,
                cache=ModelCostCache(read=0, write=0),
            ),
            limit=ModelLimit(
                context=m.get("model_info", {}).get("llama.context_length", 4096),
                output=m.get("model_info", {}).get("llama.vocab_size", 0),
            ),
            status="active",
            options={},
            headers={},
            releaseDate=m.get("modified_at", "2025-01-01"),
        )
        serialized[mid] = model

    return serialized


async def _init_local_ollama() -> None:
    """Initialize local Ollama provider if instance is running."""
    global _ollama_provider, _providers_cache

    result = await _fetch_local_ollama_models()
    if result is None:
        logger.info("Local Ollama not available, skipping")
        return

    models, model_ids = result
    serialized = _serialize_local_ollama(models, model_ids)
    if not serialized:
        logger.info("No valid Ollama models found (all embedding/completion-only)")
        return

    _ollama_provider = Provider(
        id="ollama",
        name="Local Ollama",
        source="env",
        env=[],
        options={},
        models=serialized,
    )

    _providers_cache = None
    logger.info("Local Ollama provider initialized with {} models", len(serialized))


# =========================================================================
# PUBLIC API
# =========================================================================

_providers_cache: list[Provider] | None = None


def _build_providers() -> list[Provider]:
    global _providers_cache
    if _providers_cache is not None:
        return _providers_cache

    providers = _build_providers_from_data(_models_dev_data)

    # if no data from anywhere, fall back to hardcoded priority providers
    if not providers:
        for pid in PRIORITY_PROVIDERS:
            raw = _PROVIDER_DATA.get(pid)
            if raw:
                providers.append(_transform_provider(raw))

    # add local Ollama provider if available
    if _ollama_provider is not None:
        existing = next((p for p in providers if p.id == "ollama"), None)
        if existing:
            providers = [p for p in providers if p.id != "ollama"]
        providers.insert(0, _ollama_provider)

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
