"""Small dataclasses passed between layers. Frozen so they cannot mutate across
the simulator/agent boundary.

`Transition` is the row written into the replay buffer; `EpisodeMetrics` is what
`EvaluationService.rollout` returns; `StepDiagnostic` is the per-update record
that the DDPG service appends to `TrainResult.diagnostics`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class EpisodeMetrics:
    reward: float
    length: int
    coverage: float
    collisions: int


@dataclass(frozen=True)
class StepDiagnostic:
    step: int
    actor_loss: float
    critic_loss: float
    mean_q: float
    sigma: float
    episode_reward: float
    coverage: float


@dataclass
class TrainResult:
    diagnostics: list[StepDiagnostic] = field(default_factory=list)
    final_metrics: EpisodeMetrics | None = None
