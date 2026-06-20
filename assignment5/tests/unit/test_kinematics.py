"""Layer 2 — 4-test battery for the unicycle kinematics."""

from __future__ import annotations

import math

from roomba_lab.simulator.kinematics import Pose, step_unicycle

DT = 0.1
V_MAX = 0.5
W_MAX = 1.5


def test_zero_action_no_motion() -> None:
    p = Pose(1.0, 2.0, 0.5)
    out = step_unicycle(p, (0.0, 0.0), DT, V_MAX, W_MAX)
    assert out == p


def test_max_forward_advance_in_facing_direction() -> None:
    p = Pose(0.0, 0.0, 0.0)
    out = step_unicycle(p, (1.0, 0.0), DT, V_MAX, W_MAX)
    assert out.x == pytest_approx(V_MAX * DT)
    assert out.y == pytest_approx(0.0)
    assert out.theta == pytest_approx(0.0)


def test_max_turn_changes_only_theta() -> None:
    p = Pose(2.0, 3.0, 0.0)
    out = step_unicycle(p, (0.0, 1.0), DT, V_MAX, W_MAX)
    assert out.x == pytest_approx(2.0)
    assert out.y == pytest_approx(3.0)
    assert out.theta == pytest_approx(W_MAX * DT)


def test_combined_move_matches_closed_form() -> None:
    p = Pose(0.0, 0.0, math.pi / 2)
    out = step_unicycle(p, (1.0, 0.5), DT, V_MAX, W_MAX)
    assert out.x == pytest_approx(V_MAX * DT * math.cos(math.pi / 2), abs=1e-9)
    assert out.y == pytest_approx(V_MAX * DT * math.sin(math.pi / 2))
    assert out.theta == pytest_approx(math.pi / 2 + 0.5 * W_MAX * DT)


def test_angle_wraps_to_pi() -> None:
    p = Pose(0.0, 0.0, 3.0)
    out = step_unicycle(p, (0.0, 1.0), 2.0, V_MAX, W_MAX)
    assert -math.pi <= out.theta <= math.pi


def test_action_clamped_to_unit_box() -> None:
    p = Pose(0.0, 0.0, 0.0)
    out = step_unicycle(p, (5.0, -7.0), DT, V_MAX, W_MAX)
    assert out.x == pytest_approx(V_MAX * DT)
    assert out.theta == pytest_approx(-W_MAX * DT)


def pytest_approx(target: float, abs: float = 1e-6) -> float:
    import pytest as _pytest
    return _pytest.approx(target, abs=abs)
