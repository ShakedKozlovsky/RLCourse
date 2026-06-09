"""Typed structures — construction + shape contracts."""

from __future__ import annotations

import numpy as np

from proximal_lab.shared.types import (
    EpisodeMetrics,
    IterationDiagnostics,
    RolloutBatch,
    TrainResult,
)


def test_rollout_batch_construction() -> None:
    t = 4
    batch = RolloutBatch(
        observations=np.zeros((t, 17)), actions=np.zeros((t, 6)),
        log_probs_old=np.zeros(t), values=np.zeros(t),
        rewards=np.zeros(t), dones=np.zeros(t, dtype=bool),
        advantages=np.zeros(t), returns=np.zeros(t),
    )
    assert batch.observations.shape == (t, 17)
    assert batch.dones.dtype == bool


def test_episode_metrics_construction() -> None:
    m = EpisodeMetrics(episode_index=0, total_reward=1.5, length=200, actions_mean_abs=0.3)
    assert m.length == 200
    assert m.actions_mean_abs == 0.3


def test_iteration_diagnostics_construction() -> None:
    d = IterationDiagnostics(
        iteration=1, timestep=2048, mean_episode_reward=10.0,
        mean_kl=0.01, clip_fraction=0.05, explained_variance=0.7,
        policy_loss=-0.5, value_loss=0.2, entropy=1.0,
    )
    assert d.iteration == 1
    assert d.clip_fraction == 0.05


def test_train_result_defaults() -> None:
    r = TrainResult()
    assert r.diagnostics == []
    assert r.episode_rewards == []
    assert r.final_mean_reward == 0.0
    assert r.total_timesteps == 0
