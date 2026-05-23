"""Logger factory. Library code logs through this — no print statements."""

from __future__ import annotations

import logging
import os
import sys

_LOG_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_initialised = False


def _init_root() -> None:
    """Configure the root logger once. Idempotent so re-importing is safe."""
    global _initialised
    if _initialised:
        return
    level = os.environ.get("DQN_TRADER_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FMT))
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
    _initialised = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; configure root once on first call."""
    _init_root()
    return logging.getLogger(name)
