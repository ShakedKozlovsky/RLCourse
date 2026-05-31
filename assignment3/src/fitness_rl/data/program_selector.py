"""Pick exactly one program from the summary matching the assignment's criteria."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fitness_rl.shared.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class SelectionCriteria:
    """All knobs governing which program we pick. Loaded from configs/setup.json."""

    equipment: str = "Full Gym"
    min_program_weeks: int = 4
    max_program_weeks: int = 12
    min_time_per_workout: int = 45
    max_time_per_workout: int = 120


class ProgramSelector:
    """Filter summary rows and pick a single reproducible program."""

    def __init__(self, criteria: SelectionCriteria | None = None):
        self._criteria = criteria or SelectionCriteria()

    def pick(self, summary: pd.DataFrame) -> pd.Series:
        """Return the chosen program row. Raises if no match found."""
        c = self._criteria
        mask = (
            (summary["equipment"] == c.equipment)
            & (summary["program_length"] >= c.min_program_weeks)
            & (summary["program_length"] <= c.max_program_weeks)
            & (summary["time_per_workout"] >= c.min_time_per_workout)
            & (summary["time_per_workout"] <= c.max_time_per_workout)
        )
        matches = summary[mask].copy()
        if matches.empty:
            raise ValueError(f"No program matches criteria: {c}")
        matches = matches.sort_values("title").reset_index(drop=True)
        chosen = matches.iloc[0]
        _logger.info("selected program: %r (week=%d, time=%d)",
                     chosen["title"], int(chosen["program_length"]),
                     int(chosen["time_per_workout"]))
        return chosen

    def filter_detailed(self, detailed: pd.DataFrame, chosen_title: str) -> pd.DataFrame:
        """Keep only detailed rows whose title matches ``chosen_title``."""
        filtered = detailed[detailed["title"] == chosen_title].copy()
        if filtered.empty:
            raise ValueError(f"No detailed rows for chosen program: {chosen_title!r}")
        return filtered.reset_index(drop=True)
