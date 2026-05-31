"""Audit finding #4: three reference policies to compare a trained agent against.

Without baselines, "A2C scored 7.06" is meaningless. These three policies
let the evaluator say "A2C beat random by X, beat round-robin by Y,
beat the Kaggle program by Z".

* **RandomPolicy** — uniform over the 5 actions (the floor)
* **RoundRobinPolicy** — `PUSH, PULL, LEGS, REST, PUSH, PULL, LEGS, REST, ...`
* **KaggleProgramPolicy** — replay the action labels from the actual
  Kaggle program for this trajectory (the data-driven baseline)

Each implements ``select_action(state, recent_actions, step_idx) -> int`` so
they're interchangeable inside :class:`PolicyBenchmark` below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.shared.types import Action


class BaselinePolicy:
    """Stateless ``select_action`` interface — overridden by each baseline."""

    name: str = "baseline"

    def select_action(self, state: np.ndarray, recent_actions: Sequence[int],
                       step_idx: int) -> int:
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """Uniform sample over the 5 discrete actions (the floor)."""

    name = "random"

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng(0)

    def select_action(self, state: np.ndarray, recent_actions: Sequence[int],
                       step_idx: int) -> int:
        return int(self._rng.integers(0, Action.n()))


class RoundRobinPolicy(BaselinePolicy):
    """``PUSH, PULL, LEGS, REST`` cycle — sensible naive heuristic."""

    name = "round_robin"

    def __init__(self, cycle: tuple[int, ...] = (
        int(Action.PUSH), int(Action.PULL), int(Action.LEGS), int(Action.REST),
    )):
        if not cycle:
            raise ValueError("cycle must be non-empty")
        self._cycle = tuple(int(a) for a in cycle)

    def select_action(self, state: np.ndarray, recent_actions: Sequence[int],
                       step_idx: int) -> int:
        return self._cycle[step_idx % len(self._cycle)]


class KaggleProgramPolicy(BaselinePolicy):
    """Replay the per-day actions extracted from the chosen Kaggle program."""

    name = "kaggle_program"

    def __init__(self, actions: np.ndarray):
        if actions.ndim != 1:
            raise ValueError(f"actions must be 1-D; got shape {actions.shape}")
        self._actions = actions.astype(np.int64, copy=True)

    def select_action(self, state: np.ndarray, recent_actions: Sequence[int],
                       step_idx: int) -> int:
        # Wrap modulo trajectory length so the policy is well-defined for any
        # episode length the env throws at it.
        return int(self._actions[step_idx % self._actions.shape[0]])


@dataclass(frozen=True)
class BenchmarkResult:
    """Total reward + action distribution for one policy over one rollout."""

    name: str
    total_reward: float
    action_distribution: list[float]
    rewards: list[float]


class PolicyBenchmark:
    """Run ``BaselinePolicy``s through a :class:`WorldEnv` and return totals."""

    @staticmethod
    def run(policy: BaselinePolicy, env: WorldEnv) -> BenchmarkResult:
        state, _ = env.reset()
        rewards: list[float] = []
        recent: list[int] = []
        counts = np.zeros(Action.n(), dtype=np.int64)
        terminated = False
        while not terminated:
            action = policy.select_action(state, recent, len(recent))
            state, reward, terminated, _, _ = env.step(action)
            rewards.append(float(reward))
            recent.append(int(action))
            counts[action] += 1
        total = float(counts.sum())
        dist = (counts / total).tolist() if total > 0 else [0.0] * Action.n()
        return BenchmarkResult(
            name=policy.name,
            total_reward=float(sum(rewards)),
            action_distribution=dist,
            rewards=rewards,
        )
