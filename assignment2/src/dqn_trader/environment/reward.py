"""Reward functions for the trading environment.

Friction is already deducted inside ``Portfolio.buy/sell``, so the baseline
reward is simply normalised ΔV. The risk-adjusted variant adds a rolling
Sharpe bonus over the agent's own portfolio returns. See ``PRD_reward.md``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np


class RewardFunction:
    """Abstract base. Subclasses override ``compute``."""

    def compute(self, prev_value: float, new_value: float, initial_value: float) -> float:
        """Compute the scalar reward for one env step (abstract)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any episode-local state. No-op by default."""


class BaselineReward(RewardFunction):
    """Normalised change in portfolio value. Friction is baked into ΔV."""

    def compute(self, prev_value: float, new_value: float, initial_value: float) -> float:
        """Return normalised portfolio change: (new - prev) / initial."""
        return (new_value - prev_value) / initial_value


class RiskAdjustedReward(RewardFunction):
    """Baseline + γ · rolling Sharpe of recent portfolio simple returns."""

    def __init__(self, sharpe_gamma: float, window: int) -> None:
        if sharpe_gamma < 0:
            raise ValueError("sharpe_gamma must be non-negative")
        if window < 2:
            raise ValueError("sharpe window must be >= 2")
        self._gamma = float(sharpe_gamma)
        self._window = int(window)
        self._returns: deque[float] = deque(maxlen=self._window)

    def reset(self) -> None:
        """Clear episode-local state for the risk-adjusted variant."""
        self._returns.clear()

    def compute(self, prev_value: float, new_value: float, initial_value: float) -> float:
        """Baseline delta plus a rolling annualised Sharpe bonus."""
        delta = (new_value - prev_value) / initial_value
        ret = (new_value - prev_value) / prev_value if prev_value > 0 else 0.0
        self._returns.append(ret)
        bonus = self._gamma * _rolling_sharpe(self._returns) if len(self._returns) >= 2 else 0.0
        return delta + bonus


def _rolling_sharpe(returns: Sequence[float]) -> float:
    """Annualised Sharpe of a daily-return sequence (252 trading days)."""
    arr = np.asarray(returns, dtype=np.float64)
    sd = arr.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(np.sqrt(252.0) * arr.mean() / sd)


def build_reward(variant: str, *, sharpe_gamma: float = 1.0, window: int = 20) -> RewardFunction:
    """Factory used by the env and SDK to translate config strings to objects."""
    if variant == "baseline":
        return BaselineReward()
    if variant == "risk_adjusted":
        return RiskAdjustedReward(sharpe_gamma=sharpe_gamma, window=window)
    raise ValueError(f"Unknown reward variant: {variant!r}")
