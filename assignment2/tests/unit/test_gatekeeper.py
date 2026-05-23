"""ApiGatekeeper — budget enforcement and retry behaviour."""

from __future__ import annotations

import pytest

from dqn_trader.shared.gatekeeper import ApiGatekeeper, RateLimitExceededError

_CONFIG = {
    "requests_per_minute": 2,
    "requests_per_hour": 10,
    "retry_after_seconds": 0,
    "max_retries": 2,
    "backoff_factor": 1.0,
}


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_executes_within_budget() -> None:
    gate = ApiGatekeeper(_CONFIG, clock=_FakeClock())
    assert gate.execute(lambda x: x + 1, 41) == 42


def test_per_minute_budget_blocks() -> None:
    clock = _FakeClock()
    gate = ApiGatekeeper(_CONFIG, clock=clock)
    gate.execute(lambda: 1)
    gate.execute(lambda: 1)
    with pytest.raises(RateLimitExceededError, match="per-minute"):
        gate.execute(lambda: 1)


def test_window_eviction_restores_budget() -> None:
    clock = _FakeClock()
    gate = ApiGatekeeper(_CONFIG, clock=clock)
    gate.execute(lambda: 1)
    gate.execute(lambda: 1)
    clock.t += 61
    gate.execute(lambda: 1)  # minute window evicted


def test_retries_then_succeeds() -> None:
    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    gate = ApiGatekeeper(_CONFIG, clock=_FakeClock())
    assert gate.execute(fn) == "ok"
    assert attempts["n"] == 2


def test_retries_exhausted_raises_last() -> None:
    def boom() -> None:
        raise RuntimeError("permanent")

    gate = ApiGatekeeper(_CONFIG, clock=_FakeClock())
    with pytest.raises(RuntimeError, match="permanent"):
        gate.execute(boom)
