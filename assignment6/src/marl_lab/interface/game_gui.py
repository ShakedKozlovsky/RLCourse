"""Tkinter GUI — driven by a step() callback, no event-loop in tests.

The GUI is split into two classes:
  - ``GameGuiCore``: pure-Python state + render_to_grid (testable)
  - ``GameGuiTk``: actual Tkinter widgets (skipped on headless CI)"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from marl_lab.environment.dec_pomdp import DecPomdpEnv
from marl_lab.game.actions import Action
from marl_lab.interface.board_renderer import render

PolicyFn = Callable[[str, np.ndarray], int]   # (role, obs) → action


class GameGuiCore:
    """Headless GUI core — step the env via two policy callbacks, expose render."""

    def __init__(self, env: DecPomdpEnv, cop_policy: PolicyFn,
                 thief_policy: PolicyFn) -> None:
        self.env = env
        self.cop_policy = cop_policy
        self.thief_policy = thief_policy
        self._joint_obs: dict[str, np.ndarray] | None = None
        self._done = False
        self._steps = 0
        self._winner: str | None = None

    def reset(self, seed: int | None = None) -> None:
        self._joint_obs = self.env.reset(seed=seed)
        self._done = False
        self._steps = 0
        self._winner = None

    def step(self) -> dict:
        """One env tick. Returns {steps, done, winner, board_grid}."""
        if self._joint_obs is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            return self._snapshot()
        cop_a = self.cop_policy("cop", self._joint_obs["cop"])
        thief_a = self.thief_policy("thief", self._joint_obs["thief"])
        self._joint_obs, _, done, info = self.env.step(
            {"cop": int(cop_a), "thief": int(thief_a)})
        self._steps += 1
        self._done = done
        if done:
            self._winner = info["winner"]
        return self._snapshot()

    def _snapshot(self) -> dict:
        board = self.env.board() if self._joint_obs is not None else None
        return {
            "steps": self._steps,
            "done": self._done,
            "winner": self._winner,
            "board_grid": (render(board).tolist() if board is not None else None),
        }

    def auto_play(self, max_steps: int = 100) -> dict:
        """Run until done or ``max_steps``; return the last snapshot."""
        self.reset()
        last = self._snapshot()
        for _ in range(max_steps):
            last = self.step()
            if self._done:
                break
        return last


def make_random_policy(rng_seed: int = 0) -> PolicyFn:
    """A policy that picks a uniform-random LEGAL action — for GUI demos."""
    rng = np.random.default_rng(rng_seed)

    def policy(role: str, obs: np.ndarray) -> int:    # noqa: ARG001 — obs unused
        n_legal = 6 if role == "cop" else 5
        return int(rng.integers(0, n_legal))
    return policy


def make_stay_policy() -> PolicyFn:
    """All agents always STAY — for deterministic GUI smoke tests."""
    def policy(role: str, obs: np.ndarray) -> int:   # noqa: ARG001
        return int(Action.STAY)
    return policy
