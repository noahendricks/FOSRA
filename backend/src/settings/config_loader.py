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

from backend.src.api.schemas.config_schemas import Model
from backend.src.settings.fosra_paths import fosra_paths


def load_config() -> dict[str, Any]:
    """Load TOML config and return as dict.

    Returns empty dict if no config file found.
    """

    fd = fosra_paths

    config_file = fd.config_file

    if config_file is None:
        return {}

    with open(config_file, "rb") as f:
        config_json = tomllib.load(f)
        # TODO: serialize into model when config pydantic model is finish
        # config = Model.model_validate(config_json)

        return config_json


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
