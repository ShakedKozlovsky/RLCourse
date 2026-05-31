"""ComparisonService — variance, CV, winner selection, action distribution."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_rl.services.comparison_service import (
    ComparisonReport,
    ComparisonResult,
    ComparisonService,
)
from fitness_rl.shared.types import Action, EpisodeMetrics


def _history(rewards: list[float], action_counts: list[int] | None = None
             ) -> list[EpisodeMetrics]:
    if action_counts is None:
        action_counts = [1] * Action.n()
    counts = np.array(action_counts, dtype=np.int64)
    return [
        EpisodeMetrics(episode=i, total_reward=r, mean_entropy=0.5,
                       action_counts=counts.copy())
        for i, r in enumerate(rewards)
    ]


def test_invalid_window_pct_raises() -> None:
    with pytest.raises(ValueError):
        ComparisonService(final_window_pct=0.0)
    with pytest.raises(ValueError):
        ComparisonService(final_window_pct=1.5)


def test_empty_history_raises() -> None:
    svc = ComparisonService()
    with pytest.raises(ValueError):
        svc.compare([], _history([1.0]))
    with pytest.raises(ValueError):
        svc.compare(_history([1.0]), [])


def test_a2c_wins_when_higher_final_reward() -> None:
    r_hist = _history([1.0] * 10)
    a_hist = _history([1.0] * 7 + [5.0, 5.0, 5.0])
    result = ComparisonService().compare(r_hist, a_hist)
    assert result.winner == "a2c"


def test_reinforce_wins_when_higher_final_reward() -> None:
    r_hist = _history([1.0] * 7 + [5.0, 5.0, 5.0])
    a_hist = _history([1.0] * 10)
    result = ComparisonService().compare(r_hist, a_hist)
    assert result.winner == "reinforce"


def test_tiebreaker_uses_lower_cv() -> None:
    # Both end at mean ~5; A2C tail is constant, REINFORCE tail oscillates → A2C wins.
    r_hist = _history([1.0] * 7 + [3.0, 5.0, 7.0])  # tail std large
    a_hist = _history([1.0] * 7 + [5.0, 5.0, 5.0])  # tail std 0
    result = ComparisonService().compare(r_hist, a_hist)
    assert result.winner == "a2c"


def test_final_cv_computed_over_tail() -> None:
    r_hist = _history([10.0] * 10)
    a_hist = _history([5.0] * 10)
    result = ComparisonService(final_window_pct=0.3).compare(r_hist, a_hist)
    # Constant rewards → zero std → zero CV
    assert result.reinforce.final_cv == pytest.approx(0.0)
    assert result.a2c.final_cv == pytest.approx(0.0)
    assert result.reinforce.mean_final_reward == pytest.approx(10.0)


def test_action_distribution_sums_to_one() -> None:
    r_hist = _history([1.0, 1.0, 1.0], action_counts=[4, 1, 0, 0, 0])
    a_hist = _history([1.0, 1.0, 1.0], action_counts=[1, 1, 1, 1, 1])
    result = ComparisonService().compare(r_hist, a_hist)
    assert sum(result.reinforce.action_distribution) == pytest.approx(1.0)
    assert result.reinforce.action_distribution[0] == pytest.approx(0.8)
    assert sum(result.a2c.action_distribution) == pytest.approx(1.0)


def test_to_dict_roundtrip_keys() -> None:
    r_hist = _history([1.0, 2.0])
    a_hist = _history([3.0, 4.0])
    result = ComparisonService().compare(r_hist, a_hist)
    d = result.to_dict()
    assert set(d.keys()) == {"reinforce", "a2c", "reinforce_rewards",
                              "a2c_rewards", "winner"}
    assert isinstance(d["reinforce"], dict)
    assert "mean_final_reward" in d["reinforce"]


def test_single_episode_history_handled_without_crash() -> None:
    r_hist = _history([2.0])
    a_hist = _history([3.0])
    result = ComparisonService().compare(r_hist, a_hist)
    # std over 1 sample is 0 by our fallback
    assert result.reinforce.overall_std == 0.0
    assert result.a2c.overall_std == 0.0


def test_comparison_report_serializable_via_dataclass() -> None:
    report = ComparisonReport(
        name="reinforce", n_episodes=10, mean_final_reward=1.0,
        overall_std=0.1, final_std=0.05, final_cv=0.05,
        action_distribution=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    assert report.name == "reinforce"


def test_winner_is_tie_when_means_and_cvs_equal() -> None:
    r_hist = _history([1.0] * 10)
    a_hist = _history([1.0] * 10)
    result = ComparisonService().compare(r_hist, a_hist)
    assert result.winner == "tie"


def test_comparison_result_dataclass_construction() -> None:
    """Sanity: ComparisonResult holds the rewards arrays for plotting."""
    r_hist = _history([1.0, 2.0, 3.0])
    a_hist = _history([1.5, 2.5, 3.5])
    result: ComparisonResult = ComparisonService().compare(r_hist, a_hist)
    assert result.reinforce_rewards == [1.0, 2.0, 3.0]
    assert result.a2c_rewards == [1.5, 2.5, 3.5]
