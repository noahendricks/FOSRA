"""TOML configuration loader for FOSRA.

Loads config from:
1. Explicit path argument
2. FOSRA_CONFIG environment variable
3. ./config.toml (project root)
4. ~/.config/fosra/config.toml

Secrets (API keys) stay in environment variables only.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def find_config_path(explicit_path: str | None = None) -> Path | None:
    """Find the config file path using the priority order."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        return None

    env_path = os.environ.get("FOSRA_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return cwd_config

    home_config = Path.home() / ".config" / "fosra" / "config.toml"
    if home_config.exists():
        return home_config

    return None


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load TOML config and return as dict.

    Returns empty dict if no config file found.
    """
    config_path = find_config_path(path)
    if config_path is None:
        return {}

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def merge_toml_into_settings(toml_config: dict[str, Any]) -> dict[str, Any]:
    """Convert TOML flat keys to nested env-style keys.

    pydantic-settings expects env vars like:
      - DATABASES__POSTGRES_URL for databases.postgres_url
      - MODELS__OPS__QUERY_EXPANSION for models.ops.query_expansion

    Returns a flat dict with double-underscore delimited keys.
    """
    flat: dict[str, Any] = {}

    def flatten(prefix: str, obj: dict[str, Any]) -> None:
        for key, value in obj.items():
            full_key = f"{prefix}__{key}" if prefix else key
            if isinstance(value, dict):
                flatten(full_key, value)
            else:
                flat[full_key.upper()] = value

    flatten("", toml_config)
    return flat
