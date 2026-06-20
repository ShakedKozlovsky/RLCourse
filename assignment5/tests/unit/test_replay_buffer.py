"""Layer 5 — Replay buffer tests."""

from __future__ import annotations

import numpy as np
import pytest

from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.shared.types import Transition


def _trans(i: int) -> Transition:
    return Transition(
        state=np.full(4, float(i), dtype=np.float32),
        action=np.full(2, float(i), dtype=np.float32),
        reward=float(i),
        next_state=np.full(4, float(i) + 0.5, dtype=np.float32),
        done=(i % 3 == 0),
    )


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, obs_dim=4, action_dim=2)


def test_buffer_starts_empty() -> None:
    b = ReplayBuffer(capacity=5, obs_dim=4, action_dim=2)
    assert len(b) == 0


def test_push_increments_size_up_to_capacity() -> None:
    b = ReplayBuffer(capacity=5, obs_dim=4, action_dim=2)
    for i in range(10):
        b.push(_trans(i))
    assert len(b) == 5


def test_buffer_wraps_around() -> None:
    b = ReplayBuffer(capacity=3, obs_dim=4, action_dim=2)
    for i in range(7):
        b.push(_trans(i))
    sampled_rewards = {b.rewards[0], b.rewards[1], b.rewards[2]}
    assert sampled_rewards == {4.0, 5.0, 6.0}


def test_sample_shapes() -> None:
    b = ReplayBuffer(capacity=10, obs_dim=4, action_dim=2,
                      rng=np.random.default_rng(0))
    for i in range(10):
        b.push(_trans(i))
    batch = b.sample(4)
    assert batch["state"].shape == (4, 4)
    assert batch["action"].shape == (4, 2)
    assert batch["reward"].shape == (4,)
    assert batch["next_state"].shape == (4, 4)
    assert batch["done"].shape == (4,)


def test_sample_too_large_raises() -> None:
    b = ReplayBuffer(capacity=10, obs_dim=4, action_dim=2)
    b.push(_trans(0))
    with pytest.raises(ValueError):
        b.sample(5)


def test_same_rng_same_indices() -> None:
    b1 = ReplayBuffer(capacity=10, obs_dim=4, action_dim=2,
                       rng=np.random.default_rng(123))
    b2 = ReplayBuffer(capacity=10, obs_dim=4, action_dim=2,
                       rng=np.random.default_rng(123))
    for i in range(10):
        b1.push(_trans(i))
        b2.push(_trans(i))
    a = b1.sample(4)["reward"]
    b = b2.sample(4)["reward"]
    np.testing.assert_array_equal(a, b)
