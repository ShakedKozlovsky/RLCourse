"""MCP token authorisation registry.

Tokens are loaded from the ``MARL_MCP_ALLOWED_TOKENS`` env var (comma-separated)
or programmatically. A token is just an opaque string; we compare with
``hmac.compare_digest`` to prevent timing attacks."""

from __future__ import annotations

import hmac
import os
from collections.abc import Iterable

ENV_VAR = "MARL_MCP_ALLOWED_TOKENS"


class TokenRegistry:
    """In-memory token allowlist with constant-time comparison."""

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        seed = list(tokens) if tokens is not None else self._from_env()
        self._tokens: tuple[str, ...] = tuple(t.strip() for t in seed if t.strip())

    @staticmethod
    def _from_env() -> list[str]:
        raw = os.environ.get(ENV_VAR, "")
        return [t for t in raw.split(",") if t.strip()]

    def __len__(self) -> int:
        return len(self._tokens)

    def is_authorised(self, token: str | None) -> bool:
        """Constant-time comparison against the allowlist."""
        if not token or not self._tokens:
            return False
        return any(hmac.compare_digest(token, t) for t in self._tokens)
