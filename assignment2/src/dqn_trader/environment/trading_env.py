"""Gymnasium-style trading environment.

State assembly (PRD §5):
    cols 0..7 — pre-scaled market features (8 channels × 30 days)
    col   8   — current position ∈ {0, 1} broadcast over time
    col   9   — current scaled unrealised PnL broadcast over time

step() advances one bar, executes the trade at the new bar's Close, then
marks the portfolio to that same Close. See ADR-008.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from dqn_trader.environment.portfolio import Portfolio
from dqn_trader.environment.reward import RewardFunction, build_reward
from dqn_trader.shared.types import Action, SliceData, StepInfo

N_MARKET = 8
N_PORTFOLIO = 2


class TradingEnv:
    """All-in / all-out single-asset env. Window is set by SliceData layout."""

    def __init__(
        self,
        slice_data: SliceData,
        initial_capital: float,
        alpha: float,
        beta: float,
        reward: RewardFunction | str = "baseline",
        sharpe_gamma: float = 1.0,
        sharpe_window: int = 20,
        invalid_action_penalty: float = 0.0,
    ):
        self._slice = slice_data
        self._window_size = slice_data.features.shape[1]
        if slice_data.features.shape[2] != N_MARKET:
            raise ValueError(f"Expected {N_MARKET} market channels, got {slice_data.features.shape[2]}")
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._portfolio = Portfolio(initial_capital, alpha, beta)
        self._reward: RewardFunction = (
            reward
            if isinstance(reward, RewardFunction)
            else build_reward(reward, sharpe_gamma=sharpe_gamma, window=sharpe_window)
        )
        self._invalid_penalty = float(invalid_action_penalty)
        self._step_idx = 0
        self._n_steps = slice_data.features.shape[0]

    @property
    def n_steps(self) -> int:
        """Total number of environment steps in this slice."""
        return self._n_steps

    @property
    def observation_shape(self) -> tuple[int, int]:
        """Shape of a single observation: (window_size, n_channels)."""
        return (self._window_size, N_MARKET + N_PORTFOLIO)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset portfolio and cursor to step 0; return initial observation."""
        self._portfolio = Portfolio(
            self._portfolio.initial_capital,
            self._alpha,
            self._beta,
        )
        self._reward.reset()
        self._step_idx = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one bar, execute the trade, compute reward, return the Gymnasium 5-tuple."""
        if self._step_idx >= self._n_steps - 1:
            raise RuntimeError("step() called on terminated env; reset() first")
        prev_price = self._current_price()
        prev_value = self._portfolio.value(prev_price)
        invalid = self._would_be_invalid(action)
        self._step_idx += 1
        new_price = self._current_price()
        trade = self._execute(Action(action), new_price)
        new_value = self._portfolio.value(new_price)
        reward = self._reward.compute(prev_value, new_value, self._portfolio.initial_capital)
        if invalid and self._invalid_penalty:
            reward -= self._invalid_penalty
        terminated = self._step_idx >= self._n_steps - 1
        info = StepInfo(
            portfolio_value=new_value,
            cash=self._portfolio.cash,
            position=self._portfolio.position,
            step_idx=self._step_idx,
            trade_executed=trade is not None,
            trade_value=abs(trade.notional) if trade else 0.0,
            realized_pnl_step=(new_value - prev_value) if trade and action == Action.SELL else 0.0,
        )
        return self._obs(), float(reward), terminated, False, asdict(info)

    def _current_price(self) -> float:
        return float(self._slice.raw_close[self._step_idx + self._window_size - 1])

    def _would_be_invalid(self, action: int) -> bool:
        pos = self._portfolio.position
        return (action == Action.BUY and pos == 1) or (action == Action.SELL and pos == 0)

    def _execute(self, action: Action, price: float):
        if action == Action.BUY:
            return self._portfolio.buy(price)
        if action == Action.SELL:
            return self._portfolio.sell(price)
        return None

    def _obs(self) -> np.ndarray:
        market = self._slice.features[self._step_idx]  # (window, 8)
        position_col = np.full((self._window_size, 1), self._portfolio.position, dtype=np.float32)
        pnl_scaled = self._portfolio.unrealized_pnl(self._current_price()) / self._portfolio.initial_capital
        pnl_col = np.full((self._window_size, 1), pnl_scaled, dtype=np.float32)
        return np.concatenate([market, position_col, pnl_col], axis=1).astype(np.float32)
