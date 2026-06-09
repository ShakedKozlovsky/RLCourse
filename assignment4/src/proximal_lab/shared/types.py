"""Typed structures shared across layers (rollouts, episode metrics, train results)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RolloutBatch:
    """One PPO rollout's worth of transitions, post-GAE."""

    observations: np.ndarray   # shape (T, obs_dim)
    actions: np.ndarray         # shape (T, action_dim)
    log_probs_old: np.ndarray   # shape (T,)
    values: np.ndarray          # shape (T,)
    rewards: np.ndarray         # shape (T,)
    dones: np.ndarray           # shape (T,) bool
    advantages: np.ndarray      # shape (T,)  — filled in after compute_gae
    returns: np.ndarray         # shape (T,)  — advantages + values


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode aggregates emitted by the env wrappers + collector."""

    episode_index: int
    total_reward: float
    length: int
    actions_mean_abs: float


@dataclass(frozen=True)
class IterationDiagnostics:
    """Per-PPO-iteration diagnostics — slide-21 stability pillars in one struct."""

    iteration: int
    timestep: int
    mean_episode_reward: float
    mean_kl: float
    clip_fraction: float
    explained_variance: float
    policy_loss: float
    value_loss: float
    entropy: float


@dataclass(frozen=True)
class TrainResult:
    """Outcome of ``PPOService.fit`` — per-iteration diagnostics for plotting + analysis."""

    diagnostics: list[IterationDiagnostics] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    final_mean_reward: float = 0.0
    total_timesteps: int = 0
