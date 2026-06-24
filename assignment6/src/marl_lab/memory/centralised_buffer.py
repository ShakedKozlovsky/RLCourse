"""Centralised replay buffer for CTDE training (PRD § 5, ADR-008).

Stores FULL EPISODES (variable-length up to max_seq_len), not per-step
transitions. QMIX target computation requires the GLOBAL STATE + per-agent
hidden states across the whole sequence, so per-step buffers (as in A5's
DDPG) would lose the recurrent context.

Storage layout: each episode is padded to ``max_seq_len`` with a ``mask``
indicating valid timesteps. On sample, returns a dict of shape (B, T, …)
tensors plus the mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl_lab.shared.types import EpisodeSequence

AGENTS = ("cop", "thief")


@dataclass
class BatchTensors:
    """Sample-time tensor bundle (numpy; trainer converts to torch).

    Shapes:
      state_seq:       (B, T, state_dim)
      next_state_seq:  (B, T, state_dim)
      obs_seq:         dict {agent: (B, T, obs_dim)}
      next_obs_seq:    dict {agent: (B, T, obs_dim)}
      action_seq:      dict {agent: (B, T)} int64
      reward_seq:      dict {agent: (B, T)} float32
      done_seq:        (B, T) float32  (1.0 on the terminal step of an episode)
      mask:            (B, T) float32  (1.0 for valid timesteps; 0.0 for padded)
    """
    state_seq: np.ndarray
    next_state_seq: np.ndarray
    obs_seq: dict[str, np.ndarray]
    next_obs_seq: dict[str, np.ndarray]
    action_seq: dict[str, np.ndarray]
    reward_seq: dict[str, np.ndarray]
    done_seq: np.ndarray
    mask: np.ndarray


class CentralisedReplayBuffer:
    """Ring buffer of episodes for CTDE.

    Capacity counts EPISODES (not transitions). Pads to ``max_seq_len`` for
    fixed-shape sampling."""

    def __init__(
        self,
        capacity: int,
        max_seq_len: int,
        state_dim: int,
        obs_dim: int,
        n_actions_per_agent: dict[str, int],
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
        self.capacity = int(capacity)
        self.max_seq_len = int(max_seq_len)
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)
        self.n_actions_per_agent = dict(n_actions_per_agent)
        self._rng = rng or np.random.default_rng(0)
        # Allocate fixed-size arrays
        self._state = np.zeros((capacity, max_seq_len, state_dim), dtype=np.float32)
        self._next_state = np.zeros_like(self._state)
        self._obs: dict[str, np.ndarray] = {
            a: np.zeros((capacity, max_seq_len, obs_dim), dtype=np.float32) for a in AGENTS
        }
        self._next_obs: dict[str, np.ndarray] = {
            a: np.zeros((capacity, max_seq_len, obs_dim), dtype=np.float32) for a in AGENTS
        }
        self._action: dict[str, np.ndarray] = {
            a: np.zeros((capacity, max_seq_len), dtype=np.int64) for a in AGENTS
        }
        self._reward: dict[str, np.ndarray] = {
            a: np.zeros((capacity, max_seq_len), dtype=np.float32) for a in AGENTS
        }
        self._done = np.zeros((capacity, max_seq_len), dtype=np.float32)
        self._mask = np.zeros((capacity, max_seq_len), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, ep: EpisodeSequence) -> None:
        """Append one episode to the ring; oldest is overwritten at capacity."""
        i = self._ptr
        # Reset all slots for this episode (padding == zeros + mask 0)
        self._state[i].fill(0)
        self._next_state[i].fill(0)
        self._done[i].fill(0)
        self._mask[i].fill(0)
        for a in AGENTS:
            self._obs[a][i].fill(0)
            self._next_obs[a][i].fill(0)
            self._action[a][i].fill(0)
            self._reward[a][i].fill(0)
        t_used = min(len(ep), self.max_seq_len)
        for t in range(t_used):
            tr = ep.transitions[t]
            self._state[i, t] = tr.global_state
            self._next_state[i, t] = tr.next_global_state
            self._done[i, t] = float(tr.done)
            self._mask[i, t] = 1.0
            for a in AGENTS:
                self._obs[a][i, t] = tr.joint_obs[a]
                self._next_obs[a][i, t] = tr.next_joint_obs[a]
                self._action[a][i, t] = int(tr.joint_action[a])
                self._reward[a][i, t] = float(tr.joint_reward[a])
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> BatchTensors:
        """Sample ``batch_size`` episodes uniformly with replacement."""
        if batch_size > self._size:
            raise ValueError(f"batch_size {batch_size} > buffer size {self._size}")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return BatchTensors(
            state_seq=self._state[idx],
            next_state_seq=self._next_state[idx],
            obs_seq={a: self._obs[a][idx] for a in AGENTS},
            next_obs_seq={a: self._next_obs[a][idx] for a in AGENTS},
            action_seq={a: self._action[a][idx] for a in AGENTS},
            reward_seq={a: self._reward[a][idx] for a in AGENTS},
            done_seq=self._done[idx],
            mask=self._mask[idx],
        )
