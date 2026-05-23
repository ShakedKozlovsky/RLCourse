"""Risk metrics — analytical sanity checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dqn_trader.services.risk_metrics import (
    BacktestMetrics,
    max_drawdown,
    sharpe_ratio,
    summarise,
    total_return,
    win_rate,
)


def test_total_return_simple_growth() -> None:
    eq = np.array([100.0, 120.0])
    assert total_return(eq) == pytest.approx(0.20, rel=1e-9)


def test_total_return_empty_or_single() -> None:
    assert total_return(np.array([])) == 0.0
    assert total_return(np.array([100.0])) == 0.0


def test_sharpe_constant_returns_is_zero() -> None:
    """All-equal returns ⇒ stdev=0 ⇒ Sharpe defined as 0."""
    assert sharpe_ratio(np.zeros(50)) == 0.0
    assert sharpe_ratio(np.full(50, 0.001)) == 0.0


def test_sharpe_positive_for_positive_mean_with_variance() -> None:
    rng = np.random.default_rng(0)
    rets = 0.001 + 0.01 * rng.normal(size=2000)
    assert sharpe_ratio(rets) > 0


def test_sharpe_analytical_value() -> None:
    """Hand-computed: rets alternating +0.01 / −0.01 has mean=0 ⇒ Sharpe=0."""
    rets = np.tile([0.01, -0.01], 100)
    assert sharpe_ratio(rets) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_known_dip() -> None:
    """Curve 100 → 120 → 90 → 110 has peak 120, trough 90 ⇒ DD = −0.25."""
    eq = np.array([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(eq) == pytest.approx(-0.25, rel=1e-9)


def test_max_drawdown_monotonic_increase_is_zero() -> None:
    assert max_drawdown(np.linspace(100, 200, 50)) == 0.0


def test_win_rate_handles_empty_and_nonempty() -> None:
    assert win_rate(np.array([])) == 0.0
    assert win_rate(np.array([10.0, -5.0, 0.0, 7.0])) == pytest.approx(0.5)


def test_summarise_returns_dataclass_with_expected_fields() -> None:
    eq = np.array([100.0, 110.0, 105.0, 108.0])
    pnls = np.array([3.0, -1.0, 2.0])
    s = summarise(eq, pnls)
    assert isinstance(s, BacktestMetrics)
    assert s.n_trades == 3
    assert s.final_value == 108.0
    assert math.isfinite(s.sharpe)
