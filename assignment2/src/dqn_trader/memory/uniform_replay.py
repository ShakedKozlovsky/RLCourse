"""Uniform (vanilla) replay buffer — the baseline against PER."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """Output of replay sampling. All tensors are numpy; the trainer moves to torch."""

    states: np.ndarray  # (B, window, features)
    actions: np.ndarray  # (B,)
    rewards: np.ndarray  # (B,)
    next_states: np.ndarray  # (B, window, features)
    dones: np.ndarray  # (B,) bool
    indices: np.ndarray  # (B,) int — used by PER for priority updates; uniform returns these too
    is_weights: np.ndarray  # (B,) float — all 1.0 for uniform replay


class UniformReplay:
    """Circular FIFO replay with uniform random sampling."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = int(capacity)
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self._cap)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._buf.append((state.astype(np.float32, copy=False), int(action), float(reward),
                          next_state.astype(np.float32, copy=False), bool(done)))

    def sample(self, batch_size: int, *, beta: float = 1.0) -> Batch:
        """Uniformly sample ``batch_size`` transitions. ``beta`` ignored (PER signature parity)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self._buf) < batch_size:
            raise ValueError(f"Not enough transitions: have {len(self._buf)}, need {batch_size}")
        idx = self._rng.integers(0, len(self._buf), size=batch_size)
        rows = [self._buf[int(i)] for i in idx]
        states = np.stack([r[0] for r in rows])
        actions = np.array([r[1] for r in rows], dtype=np.int64)
        rewards = np.array([r[2] for r in rows], dtype=np.float32)
        next_states = np.stack([r[3] for r in rows])
        dones = np.array([r[4] for r in rows], dtype=bool)
        is_weights = np.ones(batch_size, dtype=np.float32)
        _ = beta  # parity with PER signature
        return Batch(states, actions, rewards, next_states, dones, idx.astype(np.int64), is_weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """No-op for uniform replay. Kept for ReplayBuffer-interface symmetry."""
        _ = indices, td_errors
