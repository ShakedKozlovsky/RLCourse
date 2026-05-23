"""EpsilonSchedule / BetaSchedule — linear interpolation, clamping, error paths."""

from __future__ import annotations

import pytest

from dqn_trader.services.epsilon_schedule import BetaSchedule, EpsilonSchedule, LinearSchedule


def test_eps_starts_at_start_clamps_at_end() -> None:
    s = EpsilonSchedule(start=1.0, end=0.1, decay_steps=100)
    assert s.value(-5) == 1.0
    assert s.value(0) == 1.0
    assert s.value(50) == pytest.approx(0.55, rel=1e-9)
    assert s.value(100) == 0.1
    assert s.value(10_000) == 0.1


def test_beta_increasing_direction() -> None:
    b = BetaSchedule(start=0.4, end=1.0, decay_steps=100)
    assert b.value(0) == 0.4
    assert b.value(50) == pytest.approx(0.7, rel=1e-9)
    assert b.value(100) == 1.0


def test_linear_alias_is_same_class() -> None:
    assert EpsilonSchedule is LinearSchedule
    assert BetaSchedule is LinearSchedule


def test_invalid_decay_steps_raises() -> None:
    with pytest.raises(ValueError):
        EpsilonSchedule(1.0, 0.1, 0)
