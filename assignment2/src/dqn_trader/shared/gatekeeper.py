"""Rate-limited API gatekeeper.

Why: yfinance occasionally throttles or returns 429. Every yfinance call
in the codebase goes through this gatekeeper, so retries, backoff, and
budgets live in one place — the data layer never sees raw exceptions.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from dqn_trader.shared.logger import get_logger

T = TypeVar("T")

_logger = get_logger(__name__)


class RateLimitExceededError(RuntimeError):
    """Raised when the per-minute or per-hour budget is exhausted."""


class ApiGatekeeper:
    """Token-bucket gatekeeper with retries on transient failures.

    Configured by ``configs/rate_limits.json`` (passed as a Mapping).
    """

    def __init__(self, service_config: Mapping[str, Any], clock: Callable[[], float] = time.time):
        self._rpm = int(service_config["requests_per_minute"])
        self._rph = int(service_config["requests_per_hour"])
        self._retry_after = float(service_config.get("retry_after_seconds", 30))
        self._max_retries = int(service_config.get("max_retries", 3))
        self._backoff = float(service_config.get("backoff_factor", 2.0))
        self._minute_window: deque[float] = deque()
        self._hour_window: deque[float] = deque()
        self._clock = clock

    def _evict(self, now: float) -> None:
        while self._minute_window and now - self._minute_window[0] > 60:
            self._minute_window.popleft()
        while self._hour_window and now - self._hour_window[0] > 3600:
            self._hour_window.popleft()

    def _check_budget(self) -> None:
        now = self._clock()
        self._evict(now)
        if len(self._minute_window) >= self._rpm:
            raise RateLimitExceededError(f"per-minute budget exhausted ({self._rpm})")
        if len(self._hour_window) >= self._rph:
            raise RateLimitExceededError(f"per-hour budget exhausted ({self._rph})")

    def _record(self) -> None:
        now = self._clock()
        self._minute_window.append(now)
        self._hour_window.append(now)

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``fn(*args, **kwargs)`` under rate budget with retries."""
        self._check_budget()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                self._record()
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — yfinance raises a zoo of exception types
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = self._retry_after * (self._backoff**attempt)
                _logger.warning(
                    "API call failed (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc
