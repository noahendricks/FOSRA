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


class FosraPaths:
    """FOSRA directory paths, created on demand."""

    def __init__(self) -> None:
        self._home = Path.home()
        self._fosra_dir = self._home / ".fosra"
        self._data_dir = self._fosra_dir / "data"
        self._skills_dir = self._fosra_dir / "skills"
        self._config_dir = self._home / ".config" / "fosra"
        self._config_file = self._config_dir / "config.toml"
        self._ensure()

    def _ensure(self) -> None:
        created: list[str] = []
        existing: list[str] = []

        for path in [
            self._fosra_dir,
            self._config_dir,
            self._config_file,
            self._data_dir,
            self._skills_dir,
        ]:
            if path.exists():
                existing.append(str(path))
            else:
                if path in [self._fosra_dir, self.skills_dir, self.data_dir]:
                    path.mkdir(
                        parents=True,
                    )
                    created.append(str(path))
                elif path in [self._config_file]:
                    path.touch()
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
        return self._skills_dir

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def config_file(self) -> Path:
        return self._config_file

    @property
    def data_dir(self) -> Path:
        return self._data_dir


fosra_paths = FosraPaths()
