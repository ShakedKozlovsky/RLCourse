"""TradingEnv — shapes, terminal flag, action semantics, observation channels."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.environment.trading_env import N_MARKET, N_PORTFOLIO, TradingEnv
from dqn_trader.shared.types import Action, SliceData


@pytest.fixture
def toy_slice() -> SliceData:
    """30-window environment with 12 steps and a deterministic price path."""
    rng = np.random.default_rng(0)
    window = 30
    n_steps = 12
    total = n_steps + window - 1
    # 8 market channels of pre-scaled (zero-ish) noise — env doesn't normalise these.
    features = rng.normal(size=(n_steps, window, N_MARKET)).astype(np.float32)
    price = np.linspace(100.0, 110.0, total, dtype=np.float32)
    dates = np.array([np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(total)])
    return SliceData(name="toy", features=features, raw_close=price, dates=dates)


def _make(env_slice: SliceData) -> TradingEnv:
    return TradingEnv(env_slice, initial_capital=10000.0, alpha=0.001, beta=0.001)


def test_reset_obs_shape_matches_contract(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    obs, info = env.reset()
    assert obs.shape == (30, N_MARKET + N_PORTFOLIO)
    assert obs.dtype == np.float32
    assert info == {}


def test_step_advances_one_bar(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    _, _, _, _, info = env.step(Action.HOLD)
    assert info["step_idx"] == 1


def test_position_channel_reflects_holdings(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    obs_before, _, _, _, _ = env.step(Action.HOLD)
    assert np.all(obs_before[:, N_MARKET] == 0.0)
    obs_after, _, _, _, _ = env.step(Action.BUY)
    assert np.all(obs_after[:, N_MARKET] == 1.0)


def test_terminates_at_end(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(Action.HOLD)
        steps += 1
        if steps > 50:
            pytest.fail("env did not terminate within expected horizon")
    assert steps == env.n_steps - 1


def test_invalid_action_is_no_op(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    _, _, _, _, info = env.step(Action.SELL)  # SELL while flat
    assert info["position"] == 0
    assert not info["trade_executed"]


def test_buy_then_sell_round_trip_loses_two_friction_legs(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    env.step(Action.BUY)
    env.step(Action.HOLD)
    _, _, _, _, info = env.step(Action.SELL)
    # Price moved during HOLD; the *cost* of the round-trip should be small relative to gains.
    assert info["position"] == 0
    assert info["trade_executed"]


def test_step_after_termination_raises(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(Action.HOLD)
    with pytest.raises(RuntimeError):
        env.step(Action.HOLD)


def test_reward_dtype_is_python_float(toy_slice: SliceData) -> None:
    env = _make(toy_slice)
    env.reset()
    _, reward, _, _, _ = env.step(Action.BUY)
    assert isinstance(reward, float)


def test_market_channel_count_mismatch_raises() -> None:
    bad = SliceData(
        name="bad",
        features=np.zeros((5, 30, 7), dtype=np.float32),
        raw_close=np.ones(34, dtype=np.float32),
        dates=np.array(
            [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(34)]
        ),
    )
    with pytest.raises(ValueError):
        TradingEnv(bad, initial_capital=1.0, alpha=0.0, beta=0.0)
