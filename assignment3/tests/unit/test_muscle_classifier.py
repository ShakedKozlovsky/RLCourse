"""MuscleClassifier — keyword-based exercise → muscle group mapping."""

from __future__ import annotations

import pytest

from fitness_rl.data.muscle_classifier import classify
from fitness_rl.shared.types import MuscleGroup


@pytest.mark.parametrize("name,expected", [
    ("Bench Press", MuscleGroup.PUSH),
    ("Incline Dumbbell Press", MuscleGroup.PUSH),
    ("Shoulder Press", MuscleGroup.PUSH),
    ("Push-up", MuscleGroup.PUSH),
    ("Tricep Pushdown", MuscleGroup.PUSH),
    ("Barbell Row", MuscleGroup.PULL),
    ("Pull-up", MuscleGroup.PULL),
    ("Bicep Curl", MuscleGroup.PULL),
    ("Deadlift", MuscleGroup.PULL),
    ("Squat", MuscleGroup.LEGS),
    ("Leg Press", MuscleGroup.LEGS),
    ("Calf Raise", MuscleGroup.LEGS),
    ("Hip Thrust", MuscleGroup.LEGS),
    ("Plank", MuscleGroup.CORE),
    ("Crunch", MuscleGroup.CORE),
    ("Russian Twist", MuscleGroup.CORE),
    ("Banded ankle distractions", MuscleGroup.CORE),
    ("Running", MuscleGroup.CARDIO),
    ("HIIT", MuscleGroup.CARDIO),
    ("Burpee", MuscleGroup.CARDIO),
])
def test_classify_known_exercises(name: str, expected: MuscleGroup) -> None:
    assert classify(name) == expected


def test_classify_unknown_returns_unknown() -> None:
    assert classify("Quantum Yoga") == MuscleGroup.UNKNOWN


def test_classify_empty_or_nonstring_returns_unknown() -> None:
    assert classify("") == MuscleGroup.UNKNOWN
    assert classify("   ") == MuscleGroup.UNKNOWN
    assert classify(None) == MuscleGroup.UNKNOWN  # type: ignore[arg-type]


def test_classify_case_insensitive() -> None:
    assert classify("BENCH PRESS") == MuscleGroup.PUSH
    assert classify("squat") == MuscleGroup.LEGS
