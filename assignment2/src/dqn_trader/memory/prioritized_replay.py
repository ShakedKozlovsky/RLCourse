"""Proportional Prioritized Experience Replay (Schaul et al. 2016).

Sample prob ∝ ``(|δ| + ε)^α``; the bias is corrected by an importance-
sampling weight ``w_i = (1 / (N · P(i)))^β`` annealed from β_start to 1.
"""

from __future__ import annotations

import numpy as np

from dqn_trader.memory.sum_tree import SumTree, sample_indices
from dqn_trader.memory.uniform_replay import Batch


class PrioritizedReplay:
    """Capacity-N circular buffer with priority-weighted sampling."""

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        seed: int | None = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self._cap = int(capacity)
        self._alpha = float(alpha)
        self._epsilon = float(epsilon)
        self._tree = SumTree(self._cap)
        self._states: list[np.ndarray | None] = [None] * self._cap
        self._actions = np.zeros(self._cap, dtype=np.int64)
        self._rewards = np.zeros(self._cap, dtype=np.float32)
        self._next_states: list[np.ndarray | None] = [None] * self._cap
        self._dones = np.zeros(self._cap, dtype=bool)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._tree)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """New transitions enter with the current max priority — guaranteed at least one sample."""
        data_idx = self._tree.add(self._tree.max_priority)
        self._states[data_idx] = state.astype(np.float32, copy=False)
        self._actions[data_idx] = int(action)
        self._rewards[data_idx] = float(reward)
        self._next_states[data_idx] = next_state.astype(np.float32, copy=False)
        self._dones[data_idx] = bool(done)

    def sample(self, batch_size: int, *, beta: float) -> Batch:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        n = len(self)
        if n < batch_size:
            raise ValueError(f"Not enough transitions: have {n}, need {batch_size}")
        indices, priorities = sample_indices(self._tree, batch_size, self._rng)
        probs = priorities / max(self._tree.total, 1e-12)
        # IS weights normalised by the maximum weight in the batch (∈ (0, 1]).
        weights = (n * probs) ** (-beta)
        weights = weights / weights.max()
        states = np.stack([self._states[i] for i in indices])
        next_states = np.stack([self._next_states[i] for i in indices])
        return Batch(
            states=states,
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=next_states,
            dones=self._dones[indices].copy(),
            indices=indices.copy(),
            is_weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Apply ``new_priority = (|δ| + ε)^α`` to each sampled leaf."""
        if len(indices) != len(td_errors):
            raise ValueError("indices and td_errors must have the same length")
        for i, delta in zip(indices, td_errors, strict=True):
            priority = (abs(float(delta)) + self._epsilon) ** self._alpha
            self._tree.update(int(i), priority)
