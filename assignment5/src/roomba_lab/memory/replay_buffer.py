"""Fixed-capacity NumPy ring buffer for DDPG transitions.

ADR-006: store NumPy arrays (not torch tensors) — no GPU leakage, reproducible RNG.
Conversion to tensors happens at sample time, inside `services/ddpg_update.py`."""

from __future__ import annotations

import numpy as np

from roomba_lab.shared.types import Transition


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = rng or np.random.default_rng(0)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, transition: Transition) -> None:
        i = self._ptr
        self.states[i] = transition.state
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.next_states[i] = transition.next_state
        self.dones[i] = float(transition.done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > self._size:
            raise ValueError(f"batch_size {batch_size} > buffer size {self._size}")
        idx = self._rng.integers(low=0, high=self._size, size=batch_size)
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
        }
