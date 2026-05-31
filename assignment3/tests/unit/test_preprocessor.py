"""Preprocessor — negative-value handling and NaN drops."""

from __future__ import annotations

import math

import pandas as pd

from fitness_rl.data.preprocessor import Preprocessor


def test_negative_reps_under_600_kept_as_seconds() -> None:
    df = pd.DataFrame({
        "week": [1], "day": [1], "exercise_name": ["Plank"],
        "sets": [3.0], "reps": [-60.0],
    })
    out = Preprocessor().clean(df)
    assert out["reps"].iloc[0] == 60.0


def test_negative_reps_over_600_dropped() -> None:
    df = pd.DataFrame({
        "week": [1], "day": [1], "exercise_name": ["Bad"],
        "sets": [3.0], "reps": [-9999.0],
    })
    out = Preprocessor().clean(df)
    assert out.empty


def test_nan_rows_dropped() -> None:
    df = pd.DataFrame({
        "week": [1, 1], "day": [1, 2], "exercise_name": ["Good", "Bad"],
        "sets": [3.0, float("nan")], "reps": [10.0, 5.0],
    })
    out = Preprocessor().clean(df)
    assert len(out) == 1
    assert out["exercise_name"].iloc[0] == "Good"


def test_string_reps_coerced_to_numeric() -> None:
    df = pd.DataFrame({
        "week": [1], "day": [1], "exercise_name": ["Good"],
        "sets": ["3"], "reps": ["10"],
    })
    out = Preprocessor().clean(df)
    assert out["reps"].iloc[0] == 10.0
    assert math.isclose(out["sets"].iloc[0], 3.0)
