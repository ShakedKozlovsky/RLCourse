"""Typed structures shared across layers.

Why centralised: every layer (env, training, services) talks about the same
notions (Action, transition tuple, step result). Defining them once kills
silent shape/order drift and makes Python's type checker do useful work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    """Discrete trading actions. Integer values are the contract with the network."""

    SELL = 0
    HOLD = 1
    BUY = 2

    @classmethod
    def n(cls) -> int:
        """Number of actions, used by the model output head."""
        return len(cls)


@dataclass(frozen=True)
class Transition:
    """A single experience tuple stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class StepInfo:
    """Diagnostic info returned by TradingEnv.step (mirrors Gymnasium's `info`)."""

    portfolio_value: float
    cash: float
    position: int
    step_idx: int
    trade_executed: bool
    trade_value: float
    realized_pnl_step: float


@dataclass(frozen=True)
class SliceData:
    """A frozen (X, y) slice of windowed features for one of train/val/test."""

    name: str
    features: np.ndarray  # shape (N, window_size, n_features)
    raw_close: np.ndarray  # shape (N + window_size - 1,) — needed by TradingEnv to mark to market
    dates: (
        np.ndarray
    )  # shape (N + window_size - 1,) — pandas datetimes as ndarray for fast indexing
