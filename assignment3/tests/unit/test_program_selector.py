"""ProgramSelector — criteria matching and reproducible single-program pick."""

from __future__ import annotations

import pandas as pd
import pytest

from fitness_rl.data.program_selector import ProgramSelector, SelectionCriteria


def test_picks_matching_program(synthetic_summary: pd.DataFrame) -> None:
    chosen = ProgramSelector().pick(synthetic_summary)
    assert chosen["title"] == "Program Match"


def test_no_match_raises() -> None:
    df = pd.DataFrame({"title": ["X"], "equipment": ["Home Gym"],
                       "program_length": [3.0], "time_per_workout": [30.0]})
    with pytest.raises(ValueError, match="No program matches"):
        ProgramSelector().pick(df)


def test_filter_detailed_returns_only_chosen(synthetic_summary: pd.DataFrame,
                                              synthetic_detailed: pd.DataFrame) -> None:
    selector = ProgramSelector()
    chosen = selector.pick(synthetic_summary)
    filtered = selector.filter_detailed(synthetic_detailed, chosen["title"])
    assert (filtered["title"] == "Program Match").all()


def test_filter_detailed_empty_raises(synthetic_detailed: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No detailed rows"):
        ProgramSelector().filter_detailed(synthetic_detailed, "Nonexistent")


def test_reproducible_pick_with_ties(synthetic_summary: pd.DataFrame) -> None:
    """Sort by title means the same program is picked every time."""
    a = ProgramSelector().pick(synthetic_summary)
    b = ProgramSelector().pick(synthetic_summary)
    assert a["title"] == b["title"]


def test_custom_criteria_used() -> None:
    df = pd.DataFrame({
        "title": ["A"], "equipment": ["Home"],
        "program_length": [4.0], "time_per_workout": [60.0],
    })
    c = SelectionCriteria(equipment="Home", min_program_weeks=4,
                          max_program_weeks=12, min_time_per_workout=45,
                          max_time_per_workout=120)
    chosen = ProgramSelector(c).pick(df)
    assert chosen["title"] == "A"
