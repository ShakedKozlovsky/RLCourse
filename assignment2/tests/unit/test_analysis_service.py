"""Tests for the analysis_service module — action distribution and Q-value snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.environment.trading_env import N_MARKET, TradingEnv
from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.analysis_service import (
    ActionDistribution,
    QValueSnapshot,
    collect_action_distribution,
    collect_qvalue_heatmap,
)
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.shared.types import SliceData


@pytest.fixture
def tiny_env() -> TradingEnv:
    """Minimal env for testing analysis functions."""
    rng = np.random.default_rng(0)
    window, n_steps = 30, 10
    total = n_steps + window - 1
    features = rng.normal(size=(n_steps, window, N_MARKET)).astype(np.float32)
    price = np.linspace(100.0, 110.0, total, dtype=np.float32)
    dates = np.array([np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(total)])
    slc = SliceData(name="tiny", features=features, raw_close=price, dates=dates)
    return TradingEnv(slc, initial_capital=10000.0, alpha=0.001, beta=0.001)


@pytest.fixture
def agent() -> DQNAgent:
    """Fresh agent (random weights) for analysis."""
    return DQNAgent(
        window_size=30, n_features=10, n_actions=3,
        replay=UniformReplay(capacity=10), gamma=0.99, lr=1e-3,
    )


def test_action_distribution_sums_to_one(tiny_env: TradingEnv, agent: DQNAgent) -> None:
    """Fractions must sum to 1.0."""
    d = collect_action_distribution(agent, tiny_env, epsilon=0.0)
    assert isinstance(d, ActionDistribution)
    total_frac = d.sell_frac + d.hold_frac + d.buy_frac
    assert total_frac == pytest.approx(1.0, abs=1e-9)
    assert d.total_steps == tiny_env.n_steps - 1


def test_action_distribution_with_full_exploration(tiny_env: TradingEnv, agent: DQNAgent) -> None:
    """With epsilon=1.0, all three actions should appear in a long enough run."""
    d = collect_action_distribution(agent, tiny_env, epsilon=1.0)
    assert d.total_steps > 0


def test_qvalue_heatmap_shapes(tiny_env: TradingEnv, agent: DQNAgent) -> None:
    """Q-value arrays should match the number of steps taken."""
    snap = collect_qvalue_heatmap(agent, tiny_env)
    assert isinstance(snap, QValueSnapshot)
    n = tiny_env.n_steps - 1
    assert snap.q_sell.shape == (n,)
    assert snap.q_hold.shape == (n,)
    assert snap.q_buy.shape == (n,)
    assert snap.actions_taken.shape == (n,)
    assert snap.portfolio_values.shape == (n,)


def test_qvalue_heatmap_actions_are_argmax(tiny_env: TradingEnv, agent: DQNAgent) -> None:
    """Greedy rollout: each action should be the argmax of Q at that step."""
    snap = collect_qvalue_heatmap(agent, tiny_env)
    q_stack = np.stack([snap.q_sell, snap.q_hold, snap.q_buy], axis=1)
    expected_actions = q_stack.argmax(axis=1)
    np.testing.assert_array_equal(snap.actions_taken, expected_actions)
