"""Stdlib logger factory — single configuration point for the whole project.

V3 § 3.3 — no `print` calls in library code; everything goes through a named
logger that callers can attach handlers to."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger; the first call configures root output."""
    _configure_root()
    return logging.getLogger(name)
