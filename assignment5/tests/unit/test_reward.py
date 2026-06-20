"""Layer 3 — reward-function math tests (pure)."""

from __future__ import annotations

import pytest

from roomba_lab.environment.reward import RewardConfig, RewardInputs, compute_reward

CFG = RewardConfig(
    new_cell_bonus=1.0,
    collision_penalty=-10.0,
    step_penalty=-0.01,
    completion_bonus=100.0,
    coverage_target=0.85,
)

CFG_WITH_PROGRESS = RewardConfig(
    new_cell_bonus=1.0,
    collision_penalty=-1.0,
    step_penalty=-0.05,
    completion_bonus=100.0,
    coverage_target=0.30,
    coverage_progress_coef=50.0,
)


def test_no_motion_yields_step_penalty_only() -> None:
    r, info = compute_reward(RewardInputs(new_cells=0, collided=False,
                                            coverage_before=0.5,
                                            coverage_after=0.5), CFG)
    assert r == pytest.approx(-0.01)
    assert info["new_cells"] == 0
    assert info["bonus_fired"] == 0


def test_new_cells_award_bonus() -> None:
    r, _ = compute_reward(RewardInputs(new_cells=10, collided=False,
                                        coverage_before=0.5,
                                        coverage_after=0.55), CFG)
    assert r == pytest.approx(10.0 - 0.01)


def test_collision_penalises() -> None:
    r, info = compute_reward(RewardInputs(new_cells=0, collided=True,
                                            coverage_before=0.1,
                                            coverage_after=0.1), CFG)
    assert r == pytest.approx(-10.0 - 0.01)
    assert info["collided"] == 1


def test_crossing_target_fires_completion_bonus() -> None:
    r, info = compute_reward(RewardInputs(new_cells=5, collided=False,
                                            coverage_before=0.84,
                                            coverage_after=0.86), CFG)
    assert r == pytest.approx(5.0 - 0.01 + 100.0)
    assert info["bonus_fired"] == 1


def test_completion_only_fires_when_crossing() -> None:
    r, info = compute_reward(RewardInputs(new_cells=0, collided=False,
                                            coverage_before=0.86,
                                            coverage_after=0.87), CFG)
    assert info["bonus_fired"] == 0
    assert r == pytest.approx(-0.01)


def test_coverage_progress_shaping_adds_dense_signal() -> None:
    """The tuned reward includes a dense `coverage_progress_coef × Δcoverage`
    term — fixes the over-weighted collision-avoidance discovered at 20k steps."""
    r, _ = compute_reward(RewardInputs(new_cells=0, collided=False,
                                         coverage_before=0.10,
                                         coverage_after=0.12),
                           CFG_WITH_PROGRESS)
    expected = -0.05 + 50.0 * 0.02
    assert r == pytest.approx(expected)


def test_coverage_progress_clamped_to_non_negative() -> None:
    """Coverage can only grow (cells never become un-visited); the term is
    one-sided to avoid punishing reset transitions."""
    r, _ = compute_reward(RewardInputs(new_cells=0, collided=False,
                                         coverage_before=0.20,
                                         coverage_after=0.0),
                           CFG_WITH_PROGRESS)
    assert r == pytest.approx(-0.05)
