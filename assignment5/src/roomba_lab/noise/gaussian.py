"""Independent Gaussian exploration noise (the default per ADR-005).

Slide 7 of L09: `a_t = clip(μ(s_t) + N_t, −1, +1)` with `N_t ~ Normal(0, σ²I)`."""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    def __init__(
        self,
        action_dim: int,
        sigma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self.action_dim = int(action_dim)
        self.sigma = float(sigma)
        self._rng = rng or np.random.default_rng(0)

    def set_sigma(self, sigma: float) -> None:
        """Set sigma."""
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)

    def sample(self) -> np.ndarray:
        """Sample."""
        return self._rng.normal(loc=0.0, scale=self.sigma,
                                 size=(self.action_dim,)).astype(np.float32)

    def reset(self) -> None:
        """Gaussian is memoryless; provided for API symmetry with OUNoise."""
