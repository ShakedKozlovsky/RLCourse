"""BacktestService — run a trained policy on a SliceData under greedy decisions.

Reports a BacktestMetrics summary, the full equity curve, the Buy-and-Hold
benchmark (same slice), and an action log. Pure evaluation — no learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dqn_trader.environment.trading_env import TradingEnv
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.risk_metrics import BacktestMetrics, summarise
from dqn_trader.shared.types import Action, SliceData


@dataclass(frozen=True)
class BacktestResult:
    """Backtest payload — metrics + curves + the per-step action sequence."""

    metrics: BacktestMetrics
    equity: np.ndarray
    benchmark: np.ndarray
    actions: np.ndarray
    trade_pnls: np.ndarray


class BacktestService:
    """Evaluates a trained agent on a single slice. Greedy policy, no exploration."""

    def __init__(self, env: TradingEnv, agent: DQNAgent):
        self._env = env
        self._agent = agent

    def run(self) -> BacktestResult:
        state, _ = self._env.reset()
        equity = [self._initial_value()]
        actions: list[int] = []
        trade_pnls: list[float] = []
        last_buy_value: float | None = None
        rng = np.random.default_rng(0)
        done = False
        while not done:
            action = self._agent.act(state, epsilon=0.0, rng=rng)
            next_state, _, done, _, info = self._env.step(action)
            equity.append(info["portfolio_value"])
            actions.append(action)
            if info["trade_executed"]:
                if action == Action.BUY:
                    last_buy_value = info["portfolio_value"]
                elif action == Action.SELL and last_buy_value is not None:
                    trade_pnls.append(info["portfolio_value"] - last_buy_value)
                    last_buy_value = None
            state = next_state
        eq = np.asarray(equity, dtype=np.float64)
        pnls = np.asarray(trade_pnls, dtype=np.float64)
        return BacktestResult(
            metrics=summarise(eq, pnls),
            equity=eq,
            benchmark=self._benchmark_curve(),
            actions=np.asarray(actions, dtype=np.int64),
            trade_pnls=pnls,
        )

    def _initial_value(self) -> float:
        return self._env._portfolio.initial_capital  # type: ignore[attr-defined]

    def _benchmark_curve(self) -> np.ndarray:
        """Buy-and-Hold on day 0 of the slice's *evaluation window*."""
        slc: SliceData = self._env._slice  # type: ignore[attr-defined]
        window = self._env.observation_shape[0]
        prices = slc.raw_close[window - 1 :]  # one price per env step (incl. step 0)
        v0 = self._initial_value()
        return (v0 * prices / prices[0]).astype(np.float64)


def save_backtest(result: BacktestResult, out_dir: Path, name: str) -> Path:
    """Write JSON metrics + numpy arrays for a single backtest run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{name}.npz"
    np.savez_compressed(
        npz,
        equity=result.equity,
        benchmark=result.benchmark,
        actions=result.actions,
        trade_pnls=result.trade_pnls,
    )
    import json

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(result.metrics.__dict__, indent=2))
    return npz
