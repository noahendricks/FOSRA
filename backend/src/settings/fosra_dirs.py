"""
FOSRA directory layout.

Standard locations:
  ~/.fosra/              — user-level FOSRA directory (skills, data)
  ~/.config/fosra/       — config directory
  ~/.config/fosra/config.toml  — configuration file

On first startup these are created automatically if they don't exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from loguru import logger


class FosraDirs:
    """FOSRA directory paths, created on demand."""

    def __init__(self) -> None:
        self._home = Path.home()
        self._fosra_dir = self._home / ".fosra"
        self._config_dir = self._home / ".config" / "fosra"
        self._ensure()

    def _ensure(self) -> None:
        created: list[str] = []
        existing: list[str] = []
        for path in [self._fosra_dir, self._config_dir]:
            if path.exists():
                existing.append(str(path))
            else:
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))
        if created:
            logger.info(f"FOSRA directories created: {created}")
        if existing:
            logger.debug(f"FOSRA directories already exist: {existing}")

    @property
    def fosra(self) -> Path:
        return self._fosra_dir

    @property
    def skills_dir(self) -> Path:
        return self._fosra_dir / "skills"

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def config_path(self) -> Path:
        return self._config_dir / "config.toml"

    @property
    def data_dir(self) -> Path:
        return self._fosra_dir / "data"


fosra_dirs = FosraDirs()
