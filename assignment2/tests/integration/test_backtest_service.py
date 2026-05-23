"""BacktestService — end-to-end on a tiny env with a freshly initialised agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dqn_trader.environment.trading_env import N_MARKET, TradingEnv
from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.backtest_service import BacktestService, save_backtest
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.risk_metrics import BacktestMetrics
from dqn_trader.shared.types import SliceData


@pytest.fixture
def env() -> TradingEnv:
    rng = np.random.default_rng(0)
    window = 30
    n_steps = 15
    total = n_steps + window - 1
    features = rng.normal(size=(n_steps, window, N_MARKET)).astype(np.float32)
    price = np.linspace(100.0, 110.0, total, dtype=np.float32)
    dates = np.array([np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(total)])
    slc = SliceData(name="bt", features=features, raw_close=price, dates=dates)
    return TradingEnv(slc, initial_capital=10000.0, alpha=0.001, beta=0.001)


@pytest.fixture
def agent() -> DQNAgent:
    return DQNAgent(
        window_size=30, n_features=10, n_actions=3,
        replay=UniformReplay(capacity=10), gamma=0.99, lr=1e-3,
    )


def test_backtest_runs_full_horizon(env: TradingEnv, agent: DQNAgent) -> None:
    res = BacktestService(env, agent).run()
    assert res.equity.size == env.n_steps  # initial + (n_steps - 1) step values
    assert res.benchmark.size == env.n_steps
    assert res.actions.size == env.n_steps - 1


def test_backtest_metrics_have_expected_shape(env: TradingEnv, agent: DQNAgent) -> None:
    res = BacktestService(env, agent).run()
    assert isinstance(res.metrics, BacktestMetrics)
    assert np.isfinite(res.metrics.total_return)
    assert np.isfinite(res.metrics.sharpe)


def test_save_writes_npz_and_json(env: TradingEnv, agent: DQNAgent, tmp_path: Path) -> None:
    res = BacktestService(env, agent).run()
    npz_path = save_backtest(res, tmp_path, name="run0")
    assert npz_path.exists()
    assert (tmp_path / "run0.json").exists()
    data = np.load(npz_path)
    assert "equity" in data.files


def test_benchmark_starts_at_initial_capital(env: TradingEnv, agent: DQNAgent) -> None:
    res = BacktestService(env, agent).run()
    assert res.benchmark[0] == pytest.approx(env._portfolio.initial_capital, rel=1e-9)  # type: ignore[attr-defined]
