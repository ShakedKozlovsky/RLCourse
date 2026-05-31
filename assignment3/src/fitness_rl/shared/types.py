"""Typed structures shared across layers.

Defining these once kills silent shape/order drift between data, env,
model, and services. Every module that talks about an action or a
state references the types here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    """Discrete workout actions. Integer values are the contract with the network."""

    PUSH = 0
    PULL = 1
    LEGS = 2
    CARDIO = 3
    REST = 4

    @classmethod
    def n(cls) -> int:
        """Number of actions — used by policy heads to size the output."""
        return len(cls)


class MuscleGroup(IntEnum):
    """Coarse muscle groups inferred from exercise names. 5 groups + unknown."""

    PUSH = 0  # chest, shoulders, triceps
    PULL = 1  # back, biceps
    LEGS = 2  # legs, glutes
    CORE = 3  # abs, lower back, mobility, core
    CARDIO = 4  # cardio, conditioning
    UNKNOWN = 5

    @classmethod
    def n_meaningful(cls) -> int:
        """Count excluding UNKNOWN — for muscle-distribution vectors."""
        return 5


@dataclass(frozen=True)
class DailyStep:
    """One day in the synthetic trainee trajectory."""

    week: int
    day: int
    total_volume: float
    muscle_distribution: np.ndarray  # shape (5,) — normalized over PUSH..CARDIO
    session_duration: float  # minutes
    is_rest: bool
    dominant_muscle: MuscleGroup  # used to label per-day action for the LSTM


@dataclass(frozen=True)
class EpisodeMetrics:
    """One row in the per-episode training log."""

    episode: int
    total_reward: float
    mean_entropy: float
    action_counts: np.ndarray  # shape (5,) — how many times each action was taken
