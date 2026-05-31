"""Build the 16-dim state vector per day from a trajectory.

State layout (16 dims):
    [0]    volume_normalized            = total_volume / V_max
    [1:6]  muscle_distribution (5 dims) = normalized over PUSH..CARDIO
    [6]    session_duration_norm        = session_duration / 120
    [7]    week_index_norm              = week / n_weeks
    [8:15] day_in_cycle (7-dim one-hot) = day-of-week indicator
    [15]   rest_indicator               = 1.0 if rest day else 0.0
"""

from __future__ import annotations

import numpy as np

from fitness_rl.shared.types import DailyStep

STATE_DIM = 16


class FeatureEngineer:
    """Turn a list of DailyStep into a (T, 16) float32 array."""

    def __init__(self, n_weeks: int, max_session_minutes: float = 120.0):
        if n_weeks <= 0:
            raise ValueError("n_weeks must be > 0")
        self._n_weeks = int(n_weeks)
        self._max_session = float(max_session_minutes)

    def transform(self, trajectory: list[DailyStep]) -> np.ndarray:
        """Return shape (T, 16) float32, where T = len(trajectory)."""
        if not trajectory:
            raise ValueError("trajectory must be non-empty")
        v_max = max(step.total_volume for step in trajectory) or 1.0
        out = np.zeros((len(trajectory), STATE_DIM), dtype=np.float32)
        for t, step in enumerate(trajectory):
            out[t] = self._encode_step(step, v_max)
        return out

    def _encode_step(self, step: DailyStep, v_max: float) -> np.ndarray:
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        vec[0] = step.total_volume / v_max
        vec[1:6] = step.muscle_distribution
        vec[6] = min(step.session_duration / self._max_session, 1.0)
        vec[7] = step.week / self._n_weeks
        day_idx = max(0, min(step.day - 1, 6))  # day is 1..7
        vec[8 + day_idx] = 1.0
        vec[15] = 1.0 if step.is_rest else 0.0
        return vec
