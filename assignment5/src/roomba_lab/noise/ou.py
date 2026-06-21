"""Ornstein-Uhlenbeck noise — Lillicrap 2016 original choice for DDPG.

Provided for pedagogical completeness (PRD_exploration_noise § 2.2). Selectable
via `config.noise.kind = "ou"`."""

from __future__ import annotations

import numpy as np


class OUNoise:
    def __init__(
        self,
        action_dim: int,
        theta: float,
        mu: float,
        sigma: float,
        dt: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if theta < 0.0:
            raise ValueError("theta must be >= 0")
        self.action_dim = int(action_dim)
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self._rng = rng or np.random.default_rng(0)
        self._state = np.full((action_dim,), mu, dtype=np.float32)

    def set_sigma(self, sigma: float) -> None:
        """Set sigma."""
        self.sigma = float(sigma)

    def sample(self) -> np.ndarray:
        """Sample."""
        drift = self.theta * (self.mu - self._state) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * self._rng.standard_normal(
            size=(self.action_dim,)
        )
        self._state = (self._state + drift + diffusion).astype(np.float32)
        return self._state.copy()

    def reset(self) -> None:
        """Reset."""
        self._state = np.full((self.action_dim,), self.mu, dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        """Public read-only view of the internal OU process state.

        Returns a copy so callers cannot mutate the internal state."""
        return self._state.copy()
