"""SumTree — invariants, propagation, and proportional sampling."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.memory.sum_tree import SumTree, sample_indices


def test_empty_tree_total_zero() -> None:
    assert SumTree(capacity=4).total == 0.0


def test_add_updates_total_correctly() -> None:
    t = SumTree(capacity=4)
    t.add(1.0)
    t.add(2.0)
    t.add(3.0)
    assert t.total == 6.0


def test_update_propagates_to_root() -> None:
    t = SumTree(capacity=4)
    for p in (1.0, 2.0, 3.0, 4.0):
        t.add(p)
    t.update(1, 10.0)  # was 2.0
    assert t.total == 1.0 + 10.0 + 3.0 + 4.0


def test_get_returns_leaf_covering_value() -> None:
    t = SumTree(capacity=4)
    for p in (1.0, 2.0, 3.0, 4.0):
        t.add(p)
    # Cumulative: [1, 3, 6, 10]; value=0.5 ⇒ leaf 0; value=4.0 ⇒ leaf 2; value=9.9 ⇒ leaf 3.
    assert t.get(0.5)[0] == 0
    assert t.get(4.0)[0] == 2
    assert t.get(9.9)[0] == 3


def test_sample_frequencies_match_priorities() -> None:
    """With priorities [1, 1, 1, 7] the last leaf should dominate sampling."""
    t = SumTree(capacity=4)
    for p in (1.0, 1.0, 1.0, 7.0):
        t.add(p)
    rng = np.random.default_rng(0)
    indices, _ = sample_indices(t, batch_size=400, rng=rng)
    counts = np.bincount(indices, minlength=4)
    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]


def test_max_priority_tracks_largest_inserted() -> None:
    t = SumTree(capacity=4)
    t.add(0.5)
    assert t.max_priority == 1.0  # constructor default
    t.add(2.5)
    assert t.max_priority == 2.5


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        SumTree(capacity=0)
    t = SumTree(capacity=2)
    with pytest.raises(IndexError):
        t.update(5, 1.0)
    with pytest.raises(RuntimeError):
        sample_indices(t, batch_size=1, rng=np.random.default_rng(0))
