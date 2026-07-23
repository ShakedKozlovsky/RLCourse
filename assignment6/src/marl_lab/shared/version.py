"""Single source of truth for the project version. Read by ConfigManager to
enforce that any `configs/*.yaml` whose `version` field disagrees raises early."""

from __future__ import annotations

__version__ = "1.17"
