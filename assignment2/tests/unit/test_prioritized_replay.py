"""PrioritizedReplay — IS weights, biased sampling, update path."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.memory.prioritized_replay import PrioritizedReplay


def _state(value: float = 0.0) -> np.ndarray:
    return np.full((30, 10), value, dtype=np.float32)


def _fill(buf: PrioritizedReplay, n: int) -> None:
    for i in range(n):
        buf.add(_state(float(i)), action=i % 3, reward=0.1 * i, next_state=_state(float(i + 1)), done=False)


def test_add_increments_length() -> None:
    buf = PrioritizedReplay(capacity=8, seed=0)
    _fill(buf, 5)
    assert len(buf) == 5


def test_sample_has_unit_normalised_is_weights() -> None:
    buf = PrioritizedReplay(capacity=16, seed=0)
    _fill(buf, 16)
    batch = buf.sample(batch_size=8, beta=0.4)
    assert batch.is_weights.max() == pytest.approx(1.0)
    assert (batch.is_weights > 0).all()


def test_priority_update_changes_distribution() -> None:
    """Boosting one transition's priority should make it sampled more often after."""
    buf = PrioritizedReplay(capacity=64, alpha=1.0, seed=0)
    _fill(buf, 64)
    # Drive the priority of transition 17 to a very large value.
    buf.update_priorities(np.array([17]), np.array([1000.0]))
    counts = np.zeros(64, dtype=int)
    for _ in range(20):
        batch = buf.sample(batch_size=32, beta=0.4)
        counts += np.bincount(batch.indices, minlength=64)
    assert counts[17] > counts.mean()


def test_sample_with_insufficient_data_raises() -> None:
    buf = PrioritizedReplay(capacity=10, seed=0)
    _fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(batch_size=5, beta=0.4)


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        PrioritizedReplay(capacity=10, alpha=-0.1)
    with pytest.raises(ValueError):
        PrioritizedReplay(capacity=10, epsilon=0.0)


def test_invalid_beta_raises() -> None:
    buf = PrioritizedReplay(capacity=10, seed=0)
    _fill(buf, 10)
    with pytest.raises(ValueError):
        buf.sample(batch_size=4, beta=-0.1)


def test_update_priorities_length_mismatch_raises() -> None:
    buf = PrioritizedReplay(capacity=10, seed=0)
    _fill(buf, 5)
    with pytest.raises(ValueError):
        buf.update_priorities(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_batch_state_shapes() -> None:
    buf = PrioritizedReplay(capacity=32, seed=0)
    _fill(buf, 32)
    batch = buf.sample(batch_size=8, beta=0.5)
    assert batch.states.shape == (8, 30, 10)
    assert batch.next_states.shape == (8, 30, 10)
    assert batch.actions.shape == (8,)
