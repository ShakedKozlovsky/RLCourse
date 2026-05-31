"""Aggregate detailed exercise rows into a per-day trajectory.

Each output row represents one day of the synthetic trainee — total volume,
muscle distribution across 5 groups, session duration, and a dominant-muscle
label used to infer the per-day action for LSTM training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fitness_rl.data.muscle_classifier import classify
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.types import DailyStep, MuscleGroup

_logger = get_logger(__name__)


class TrajectoryBuilder:
    """Convert detailed exercise rows for ONE program into a list of DailyStep."""

    def build(self, detailed: pd.DataFrame, n_weeks: int) -> list[DailyStep]:
        """Return one DailyStep per (week, day) — Rest Day if no exercises that day."""
        annotated = self._annotate_muscle(detailed)
        steps: list[DailyStep] = []
        for week in range(1, int(n_weeks) + 1):
            for day in range(1, 8):
                day_rows = annotated[(annotated["week"] == week) & (annotated["day"] == day)]
                steps.append(self._build_day(week, day, day_rows))
        _logger.info("trajectory built: %d days (%d training + %d rest)",
                     len(steps),
                     sum(1 for s in steps if not s.is_rest),
                     sum(1 for s in steps if s.is_rest))
        return steps

    @staticmethod
    def _annotate_muscle(detailed: pd.DataFrame) -> pd.DataFrame:
        df = detailed.copy()
        df["muscle"] = df["exercise_name"].apply(classify)
        df["volume"] = df["sets"] * df["reps"]
        return df

    @staticmethod
    def _build_day(week: int, day: int, day_rows: pd.DataFrame) -> DailyStep:
        if day_rows.empty:
            zero = np.zeros(MuscleGroup.n_meaningful(), dtype=np.float32)
            return DailyStep(week=week, day=day, total_volume=0.0,
                             muscle_distribution=zero, session_duration=0.0,
                             is_rest=True, dominant_muscle=MuscleGroup.UNKNOWN)
        total_volume = float(day_rows["volume"].sum())
        dist = np.zeros(MuscleGroup.n_meaningful(), dtype=np.float32)
        for muscle, vol in day_rows.groupby("muscle")["volume"].sum().items():
            if muscle != MuscleGroup.UNKNOWN:
                dist[int(muscle)] = float(vol)
        if dist.sum() > 0:
            dist = dist / dist.sum()
        dominant = MuscleGroup(int(dist.argmax())) if dist.sum() > 0 else MuscleGroup.UNKNOWN
        # session_duration: assume time_per_workout is consistent across rows
        duration = float(day_rows["time_per_workout"].iloc[0]) if "time_per_workout" in day_rows.columns else 0.0
        return DailyStep(week=week, day=day, total_volume=total_volume,
                         muscle_distribution=dist, session_duration=duration,
                         is_rest=False, dominant_muscle=dominant)
