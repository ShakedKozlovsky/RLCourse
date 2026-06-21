"""Configuration loader for roomba-lab. Single source of truth — no hardcoded values
in code anywhere; every magic number flows from `configs/setup.json` via this module."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roomba_lab.shared.version import __version__


def _project_root() -> Path:
    """Resolve the assignment5/ root from this file's location."""
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()
DEFAULT_SETUP_PATH = PROJECT_ROOT / "configs" / "setup.json"


class ConfigError(RuntimeError):
    """Raised on missing / malformed / version-mismatched config."""


class ConfigManager:
    """Loads JSON config and exposes it via dotted access."""

    def __init__(self, setup_path: Path | None = None) -> None:
        setup_path = setup_path or Path(
            os.environ.get("ROOMBA_LAB_CONFIG", DEFAULT_SETUP_PATH)
        )
        self._setup_path = Path(setup_path)
        self.setup: Mapping[str, Any] = self._load(self._setup_path)
        self._check_version()

    @staticmethod
    def _load(path: Path) -> Mapping[str, Any]:
        if not path.exists():
            raise ConfigError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _check_version(self) -> None:
        cfg_v = self.setup.get("version")
        if cfg_v != __version__:
            raise ConfigError(
                f"Version mismatch in setup config: config={cfg_v!r}, code={__version__!r}"
            )

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value using dotted notation, e.g. ``cfg.get('ddpg.tau')``."""
        node: Any = self.setup
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Resolve a path under ``paths.*`` against PROJECT_ROOT."""
        rel = self.get(f"paths.{key}")
        if not rel:
            raise ConfigError(f"paths.{key} missing from setup config")
        return PROJECT_ROOT / rel

    @property
    def setup_path(self) -> Path:
        """The JSON config file this manager was loaded from."""
        return self._setup_path
