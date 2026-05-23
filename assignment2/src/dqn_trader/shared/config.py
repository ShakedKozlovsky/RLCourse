"""Configuration loader. Single source of truth — no hardcoded values in code."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dqn_trader.shared.version import __version__


def _project_root() -> Path:
    """Resolve the assignment2/ root from this file's location."""
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()
DEFAULT_SETUP_PATH = PROJECT_ROOT / "configs" / "setup.json"
DEFAULT_RATE_LIMITS_PATH = PROJECT_ROOT / "configs" / "rate_limits.json"


class ConfigError(RuntimeError):
    """Raised on missing / malformed / version-mismatched config."""


class ConfigManager:
    """Loads JSON configs and exposes them as dict-like read-only access.

    Why a class (not a module-level dict): tests need to instantiate this
    against tmp paths; SDK consumers may pass in a path via env var. A class
    makes both clean without globals.
    """

    def __init__(self, setup_path: Path | None = None, rate_limits_path: Path | None = None):
        setup_path = setup_path or Path(os.environ.get("DQN_TRADER_CONFIG", DEFAULT_SETUP_PATH))
        rate_limits_path = rate_limits_path or Path(
            os.environ.get("DQN_TRADER_RATE_LIMITS", DEFAULT_RATE_LIMITS_PATH)
        )
        self._setup_path = setup_path
        self._rate_limits_path = rate_limits_path
        self.setup: Mapping[str, Any] = self._load(setup_path)
        self.rate_limits: Mapping[str, Any] = self._load(rate_limits_path)
        self._check_versions()

    @staticmethod
    def _load(path: Path) -> Mapping[str, Any]:
        if not path.exists():
            raise ConfigError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _check_versions(self) -> None:
        for name, payload in (("setup", self.setup), ("rate_limits", self.rate_limits)):
            cfg_v = payload.get("version")
            if cfg_v != __version__:
                raise ConfigError(
                    f"Version mismatch in {name}: config={cfg_v!r}, code={__version__!r}"
                )

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value using dotted notation, e.g. ``cfg.get('data.ticker')``."""
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
