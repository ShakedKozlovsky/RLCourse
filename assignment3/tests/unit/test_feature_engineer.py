"""FeatureEngineer — 16-dim state vector contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitness_rl.data.feature_engineer import STATE_DIM, FeatureEngineer
from fitness_rl.data.preprocessor import Preprocessor
from fitness_rl.data.trajectory_builder import TrajectoryBuilder


def test_state_dim_is_16(synthetic_detailed: pd.DataFrame) -> None:
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    states = FeatureEngineer(n_weeks=4).transform(steps)
    assert states.shape == (len(steps), STATE_DIM)
    assert states.dtype == np.float32


def test_volume_normalized_in_unit_interval(synthetic_detailed: pd.DataFrame) -> None:
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    states = FeatureEngineer(n_weeks=4).transform(steps)
    assert states[:, 0].min() >= 0.0
    assert states[:, 0].max() <= 1.0


def test_rest_indicator_set(synthetic_detailed: pd.DataFrame) -> None:
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    states = FeatureEngineer(n_weeks=4).transform(steps)
    for state, step in zip(states, steps, strict=True):
        expected = 1.0 if step.is_rest else 0.0
        assert state[15] == expected


def test_day_one_hot_only_one_set(synthetic_detailed: pd.DataFrame) -> None:
    df = Preprocessor().clean(synthetic_detailed)
    steps = TrajectoryBuilder().build(df, n_weeks=4)
    states = FeatureEngineer(n_weeks=4).transform(steps)
    for state in states:
        assert int(state[8:15].sum()) == 1


def test_empty_trajectory_raises() -> None:
    with pytest.raises(ValueError):
        FeatureEngineer(n_weeks=4).transform([])


def test_zero_weeks_raises() -> None:
    with pytest.raises(ValueError):
        FeatureEngineer(n_weeks=0)
