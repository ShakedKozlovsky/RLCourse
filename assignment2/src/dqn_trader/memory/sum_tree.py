"""Sum-tree data structure backing prioritized replay.

Layout (capacity = N): a single array ``tree`` of length ``2N − 1``.
Internal sums live at indices ``[0, N − 2]``; leaves at ``[N − 1, 2N − 2]``.
A data slot ``i ∈ [0, N)`` maps to leaf ``i + N − 1``. Each operation is
O(log N): update walks up to the root; sample walks down from the root.
"""

from __future__ import annotations

import numpy as np


class SumTree:
    """Fixed-capacity sum tree of float priorities."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self._size = 0
        self._write = 0
        self._max_priority = 1.0  # initial priority for new transitions

    def __len__(self) -> int:
        return self._size

    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        return self._max_priority

    def add(self, priority: float) -> int:
        """Insert at the next write slot, return the *data* index used."""
        data_idx = self._write
        self.update(data_idx, priority)
        self._write = (self._write + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        """Set leaf ``data_idx`` to ``priority`` and propagate the change up."""
        if not 0 <= data_idx < self.capacity:
            raise IndexError(f"data_idx {data_idx} out of range")
        leaf = data_idx + self.capacity - 1
        delta = priority - self._tree[leaf]
        self._tree[leaf] = priority
        node = leaf
        while node > 0:
            node = (node - 1) // 2
            self._tree[node] += delta
        if priority > self._max_priority:
            self._max_priority = float(priority)

    def get(self, value: float) -> tuple[int, float]:
        """Return ``(data_idx, priority)`` for the leaf whose prefix-sum covers ``value``."""
        if self.total <= 0:
            raise RuntimeError("Cannot sample from empty tree")
        node = 0
        while node < self.capacity - 1:
            left = 2 * node + 1
            right = left + 1
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = right
        return node - (self.capacity - 1), float(self._tree[node])


def sample_indices(tree: SumTree, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stratified PER sampling: split [0, total] into ``batch_size`` segments."""
    segment = tree.total / batch_size
    indices = np.empty(batch_size, dtype=np.int64)
    priorities = np.empty(batch_size, dtype=np.float64)
    for i in range(batch_size):
        v = rng.uniform(i * segment, (i + 1) * segment)
        idx, prio = tree.get(v)
        indices[i] = idx
        priorities[i] = prio
    return indices, priorities
