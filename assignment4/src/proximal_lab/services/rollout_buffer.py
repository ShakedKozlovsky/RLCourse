"""Rollout buffer — stores ``T × n_envs`` transitions per PPO iteration.

After the rollout completes, ``compute_advantages_and_returns`` invokes the
GAE module (Layer 4) and stores advantages + returns. ``minibatches`` yields
shuffled ``(obs, actions, log_probs, advantages, returns)`` tuples that the
PPO update consumes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size buffer for one PPO rollout — shape ``(T, n_envs, …)``."""

    def __init__(
        self,
        steps_per_rollout: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
    ):
        if steps_per_rollout < 1 or n_envs < 1:
            raise ValueError("steps_per_rollout and n_envs must be >= 1")
        if obs_dim < 1 or action_dim < 1:
            raise ValueError("obs_dim and action_dim must be >= 1")
        self.T = int(steps_per_rollout)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        shape = (self.T, self.n_envs)
        self.observations = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self._ptr = 0

    def reset(self) -> None:
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Append one transition per env. Raises if the buffer is full."""
        if self._ptr >= self.T:
            raise RuntimeError("rollout buffer full — call reset() first")
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr >= self.T

    def compute_advantages_and_returns(
        self, last_values: np.ndarray, gamma: float, lam: float
    ) -> None:
        """Run GAE once the buffer is full; store advantages + returns."""
        from proximal_lab.services.gae import compute_gae

        if not self.is_full():
            raise RuntimeError("buffer not full — fill it before computing advantages")
        for env_i in range(self.n_envs):
            self.advantages[:, env_i] = compute_gae(
                rewards=self.rewards[:, env_i],
                values=self.values[:, env_i],
                last_value=float(last_values[env_i]),
                dones=self.dones[:, env_i],
                gamma=gamma,
                lam=lam,
            )
        self.returns = self.advantages + self.values

    def minibatches(
        self, minibatch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shuffled minibatches as PyTorch tensors."""
        if minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")
        rng = rng or np.random.default_rng(0)
        total = self.T * self.n_envs
        order = rng.permutation(total)
        flat = {
            "observations": self.observations.reshape(total, self.obs_dim),
            "actions": self.actions.reshape(total, self.action_dim),
            "log_probs_old": self.log_probs.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "values_old": self.values.reshape(total),
        }
        for start in range(0, total, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield {k: torch.from_numpy(v[idx]).float() for k, v in flat.items()}
