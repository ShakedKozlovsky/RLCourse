"""SyncVectorEnv — n parallel envs in lockstep with a shared RMS."""

from __future__ import annotations

import numpy as np
import pytest

from proximal_lab.environment.vector_env import SyncVectorEnv


def test_invalid_n_envs_raises() -> None:
    with pytest.raises(ValueError):
        SyncVectorEnv("HalfCheetah-v5", n_envs=0)


def test_reset_returns_batched_obs() -> None:
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=4, seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (4, 17)
    assert obs.dtype == np.float32


def test_step_returns_batched_outputs() -> None:
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=3, seed=0)
    env.reset(seed=0)
    actions = np.stack([env.action_space.sample() for _ in range(3)])
    obs, rewards, dones, infos = env.step(actions)
    assert obs.shape == (3, 17)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert len(infos) == 3


def test_action_batch_size_mismatch_raises() -> None:
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=3, seed=0)
    env.reset(seed=0)
    bad = np.zeros((2, 6), dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(bad)


def test_shared_rms_updates_across_envs() -> None:
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    env.reset(seed=0)
    initial_count = env.shared_rms.count
    actions = np.stack([env.action_space.sample() for _ in range(2)])
    for _ in range(10):
        env.step(actions)
    assert env.shared_rms.count > initial_count
