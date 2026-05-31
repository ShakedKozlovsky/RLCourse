"""Discrete 5-action workout action space.

Defined as a tiny class (not Gymnasium's Discrete) so we don't pull
``gymnasium`` into the environment layer for one type. The model heads
size themselves off ``ActionSpace.n``.
"""

from __future__ import annotations

import numpy as np

from fitness_rl.shared.types import Action


class ActionSpace:
    """5 discrete actions: PUSH / PULL / LEGS / CARDIO / REST."""

    def __init__(self) -> None:
        self.n: int = Action.n()
        self.actions: tuple[Action, ...] = tuple(Action)

    def sample(self, rng: np.random.Generator) -> int:
        """Sample a uniformly random action index."""
        return int(rng.integers(0, self.n))

    def contains(self, action: int) -> bool:
        """Return True if ``action`` is a valid index."""
        return 0 <= int(action) < self.n

    def name(self, action: int) -> str:
        """Return the human-readable name for an action index."""
        if not self.contains(action):
            raise ValueError(f"action {action} out of range [0, {self.n})")
        return Action(int(action)).name
