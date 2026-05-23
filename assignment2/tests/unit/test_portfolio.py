"""Portfolio — buy/sell, no-ops, round-trip cost, mark-to-market math."""

from __future__ import annotations

import pytest

from dqn_trader.environment.portfolio import Portfolio


def test_initial_state_is_all_cash() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    assert p.cash == 10000.0
    assert p.shares == 0.0
    assert p.position == 0
    assert p.value(mark_price=100.0) == 10000.0


def test_buy_then_sell_round_trip_loses_two_friction_legs() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    p.buy(price=100.0)
    p.sell(price=100.0)
    expected = 10000.0 * (1.0 - 0.002) ** 2
    assert p.value(100.0) == pytest.approx(expected, rel=1e-9)


def test_buy_when_long_is_noop() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    p.buy(price=100.0)
    cash_before = p.cash
    shares_before = p.shares
    assert p.buy(price=110.0) is None
    assert p.cash == cash_before
    assert p.shares == shares_before


def test_sell_when_flat_is_noop() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    assert p.sell(price=100.0) is None
    assert p.cash == 10000.0
    assert p.position == 0


def test_mark_to_market_reflects_price_move() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    p.buy(price=100.0)
    # Value at the buy price equals cash + shares*price, which is exactly cash * (1-friction).
    v_at_buy = p.value(100.0)
    assert v_at_buy == pytest.approx(10000.0 * 0.998, rel=1e-9)
    # 10% price rise multiplies the position value by 1.10.
    v_up = p.value(110.0)
    assert v_up == pytest.approx(v_at_buy * 1.10, rel=1e-9)


def test_unrealized_pnl_zero_when_flat_and_correct_when_long() -> None:
    p = Portfolio(10000.0, 0.001, 0.001)
    assert p.unrealized_pnl(123.0) == 0.0
    p.buy(price=100.0)
    pnl = p.unrealized_pnl(105.0)
    assert pnl == pytest.approx(p.shares * 5.0, rel=1e-9)


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        Portfolio(-1.0, 0.001, 0.001)
    with pytest.raises(ValueError):
        Portfolio(100.0, -0.001, 0.0)


def test_friction_property_equals_alpha_plus_beta() -> None:
    p = Portfolio(10000.0, 0.0007, 0.0013)
    assert p.friction == pytest.approx(0.002, rel=1e-12)
