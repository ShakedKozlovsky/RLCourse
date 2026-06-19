"""Single source of truth for the project version. Read by ConfigManager (Layer 1)
to enforce that any `configs/*.json` whose `version` field disagrees raises early."""

from __future__ import annotations

__version__ = "1.00"
