"""RolloutBuffer — capacity, add/full, GAE integration, minibatches."""

from __future__ import annotations

import numpy as np
import pytest

from proximal_lab.services.rollout_buffer import RolloutBuffer


def _buf(t: int = 4, n_envs: int = 2, obs_dim: int = 3, action_dim: int = 2) -> RolloutBuffer:
    return RolloutBuffer(steps_per_rollout=t, n_envs=n_envs,
                         obs_dim=obs_dim, action_dim=action_dim)


def test_invalid_sizes_raise() -> None:
    with pytest.raises(ValueError):
        RolloutBuffer(0, 2, 3, 2)
    with pytest.raises(ValueError):
        RolloutBuffer(2, 0, 3, 2)
    with pytest.raises(ValueError):
        RolloutBuffer(2, 2, 0, 2)
    with pytest.raises(ValueError):
        RolloutBuffer(2, 2, 3, 0)


def test_add_fills_buffer() -> None:
    buf = _buf(t=3, n_envs=2)
    rng = np.random.default_rng(0)
    for _ in range(3):
        buf.add(
            obs=rng.normal(size=(2, 3)).astype(np.float32),
            action=rng.normal(size=(2, 2)).astype(np.float32),
            log_prob=rng.normal(size=2).astype(np.float32),
            value=rng.normal(size=2).astype(np.float32),
            reward=rng.normal(size=2).astype(np.float32),
            done=np.zeros(2, dtype=bool),
        )
    assert buf.is_full()


def test_add_after_full_raises() -> None:
    buf = _buf(t=2, n_envs=2)
    for _ in range(2):
        buf.add(np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool))
    with pytest.raises(RuntimeError):
        buf.add(np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool))


def test_advantages_before_full_raises() -> None:
    buf = _buf()
    with pytest.raises(RuntimeError):
        buf.compute_advantages_and_returns(np.zeros(2, dtype=np.float32),
                                             gamma=0.99, lam=0.95)


def test_compute_advantages_matches_gae_per_env() -> None:
    from proximal_lab.services.gae import compute_gae

    buf = _buf(t=3, n_envs=2)
    rng = np.random.default_rng(0)
    rewards_a, rewards_b = rng.normal(size=3), rng.normal(size=3)
    values_a, values_b = rng.normal(size=3), rng.normal(size=3)
    for step in range(3):
        buf.add(
            obs=np.zeros((2, 3), dtype=np.float32),
            action=np.zeros((2, 2), dtype=np.float32),
            log_prob=np.zeros(2, dtype=np.float32),
            value=np.array([values_a[step], values_b[step]], dtype=np.float32),
            reward=np.array([rewards_a[step], rewards_b[step]], dtype=np.float32),
            done=np.zeros(2, dtype=bool),
        )
    last_v = np.array([0.5, -0.3], dtype=np.float32)
    buf.compute_advantages_and_returns(last_v, gamma=0.99, lam=0.95)
    expected_a = compute_gae(rewards_a.astype(np.float32),
                              values_a.astype(np.float32), float(last_v[0]),
                              np.zeros(3, dtype=bool), 0.99, 0.95)
    expected_b = compute_gae(rewards_b.astype(np.float32),
                              values_b.astype(np.float32), float(last_v[1]),
                              np.zeros(3, dtype=bool), 0.99, 0.95)
    np.testing.assert_allclose(buf.advantages[:, 0], expected_a, atol=1e-5)
    np.testing.assert_allclose(buf.advantages[:, 1], expected_b, atol=1e-5)


def test_minibatches_cover_all_transitions() -> None:
    buf = _buf(t=4, n_envs=2)
    for _ in range(4):
        buf.add(np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool))
    buf.compute_advantages_and_returns(np.zeros(2, dtype=np.float32),
                                         gamma=0.99, lam=0.95)
    seen = 0
    for mb in buf.minibatches(minibatch_size=3):
        seen += mb["observations"].shape[0]
    assert seen == 4 * 2  # T × n_envs


def test_invalid_minibatch_size_raises() -> None:
    buf = _buf()
    with pytest.raises(ValueError):
        list(buf.minibatches(0))


def test_reset_allows_refill() -> None:
    buf = _buf(t=2, n_envs=2)
    for _ in range(2):
        buf.add(np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool))
    assert buf.is_full()
    buf.reset()
    assert not buf.is_full()
