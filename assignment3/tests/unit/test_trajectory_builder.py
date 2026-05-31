"""TrajectoryBuilder — day-by-day aggregation, Rest Day insertion."""

from __future__ import annotations

import pandas as pd

from fitness_rl.data.preprocessor import Preprocessor
from fitness_rl.data.trajectory_builder import TrajectoryBuilder
from fitness_rl.shared.types import MuscleGroup


def test_trajectory_has_one_step_per_day(synthetic_detailed: pd.DataFrame) -> None:
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    assert len(steps) == 4 * 7


def test_rest_days_are_marked(synthetic_detailed: pd.DataFrame) -> None:
    """Days with no exercises (days 3, 6, 7 in our fixture) must be is_rest=True."""
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    rest_day_indices = [
        i for i, s in enumerate(steps) if s.day in (3, 6, 7)
    ]
    for idx in rest_day_indices:
        assert steps[idx].is_rest, f"day {steps[idx].day} should be rest"
        assert steps[idx].total_volume == 0.0


def test_dominant_muscle_inferred(synthetic_detailed: pd.DataFrame) -> None:
    """Push-day exercises should dominate the muscle distribution."""
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    day1_steps = [s for s in steps if s.day == 1]
    assert all(s.dominant_muscle == MuscleGroup.PUSH for s in day1_steps)


def test_muscle_distribution_normalized(synthetic_detailed: pd.DataFrame) -> None:
    """For training days, distribution sums to 1; for rest days, sums to 0."""
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    for s in steps:
        total = float(s.muscle_distribution.sum())
        if s.is_rest:
            assert total == 0.0
        else:
            assert 0.99 <= total <= 1.01
