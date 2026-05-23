"""UniformReplay — add/sample/capacity wrap."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.memory.uniform_replay import UniformReplay


def _state() -> np.ndarray:
    return np.zeros((30, 10), dtype=np.float32)


def test_empty_buffer_len_zero() -> None:
    assert len(UniformReplay(capacity=10)) == 0


def test_add_increments_length() -> None:
    buf = UniformReplay(capacity=10)
    buf.add(_state(), action=1, reward=0.1, next_state=_state(), done=False)
    assert len(buf) == 1


def test_capacity_wrap() -> None:
    buf = UniformReplay(capacity=3)
    for i in range(5):
        buf.add(_state(), action=i % 3, reward=float(i), next_state=_state(), done=False)
    assert len(buf) == 3


def test_sample_returns_correct_shapes() -> None:
    buf = UniformReplay(capacity=64, seed=0)
    for _ in range(40):
        buf.add(_state(), action=1, reward=0.5, next_state=_state(), done=False)
    batch = buf.sample(batch_size=8)
    assert batch.states.shape == (8, 30, 10)
    assert batch.actions.shape == (8,)
    assert batch.is_weights.shape == (8,)
    assert (batch.is_weights == 1.0).all()


def test_sample_with_insufficient_data_raises() -> None:
    buf = UniformReplay(capacity=10)
    with pytest.raises(ValueError):
        buf.sample(batch_size=4)


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        UniformReplay(capacity=0)


def test_update_priorities_is_no_op() -> None:
    """Symmetry with PER — should silently accept and do nothing."""
    buf = UniformReplay(capacity=10, seed=0)
    for _ in range(5):
        buf.add(_state(), action=1, reward=0.0, next_state=_state(), done=False)
    buf.update_priorities(np.arange(5), np.zeros(5))
