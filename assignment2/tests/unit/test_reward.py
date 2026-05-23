"""RewardFunction variants — baseline math, factory, risk-adjusted bonus."""

from __future__ import annotations

import math

import pytest

from dqn_trader.environment.reward import (
    BaselineReward,
    RewardFunction,
    RiskAdjustedReward,
    build_reward,
)


def test_baseline_is_normalised_delta_v() -> None:
    r = BaselineReward()
    assert r.compute(10000.0, 10100.0, 10000.0) == pytest.approx(0.01, rel=1e-12)
    assert r.compute(10100.0, 10000.0, 10000.0) == pytest.approx(-0.01, rel=1e-12)


def test_baseline_zero_when_value_unchanged() -> None:
    r = BaselineReward()
    assert r.compute(10000.0, 10000.0, 10000.0) == 0.0


def test_risk_adjusted_first_step_has_no_bonus() -> None:
    """rolling Sharpe needs ≥ 2 observations; first step ⇒ bonus == 0."""
    r = RiskAdjustedReward(sharpe_gamma=1.0, window=5)
    out = r.compute(10000.0, 10100.0, 10000.0)
    assert out == pytest.approx(0.01, rel=1e-12)


def test_risk_adjusted_zero_variance_returns_only_delta() -> None:
    r = RiskAdjustedReward(sharpe_gamma=1.0, window=4)
    base = 10000.0
    for _ in range(4):
        out = r.compute(base, base, base)  # zero return ⇒ zero stdev ⇒ no bonus
        assert out == 0.0


def test_risk_adjusted_positive_with_positive_trend() -> None:
    r = RiskAdjustedReward(sharpe_gamma=1.0, window=4)
    base = 10000.0
    out_last = 0.0
    for step in range(1, 5):
        prev = base + (step - 1) * 50.0
        new = base + step * 50.0
        out_last = r.compute(prev, new, base)
    assert out_last > 0.005  # baseline ΔV/V₀ = 0.005, plus Sharpe bonus
    assert math.isfinite(out_last)


def test_build_reward_factory() -> None:
    assert isinstance(build_reward("baseline"), BaselineReward)
    assert isinstance(build_reward("risk_adjusted", sharpe_gamma=0.5, window=10), RiskAdjustedReward)
    with pytest.raises(ValueError):
        build_reward("unknown_variant")


def test_base_class_compute_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        RewardFunction().compute(1.0, 2.0, 1.0)


def test_invalid_constructions_raise() -> None:
    with pytest.raises(ValueError):
        RiskAdjustedReward(sharpe_gamma=-1.0, window=10)
    with pytest.raises(ValueError):
        RiskAdjustedReward(sharpe_gamma=1.0, window=1)
