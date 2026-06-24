"""ε-greedy action selection over discrete actions.

With probability ε take a uniform random action; otherwise argmax Q. The
discrete-action equivalent of A5's Gaussian noise on continuous actions."""

from __future__ import annotations

import numpy as np


def select_action(q_values: np.ndarray, epsilon: float,
                   rng: np.random.Generator,
                   action_mask: np.ndarray | None = None) -> int:
    """Return an action index under ε-greedy with optional masking.

    Args:
      q_values: (n_actions,) per-action Q estimates.
      epsilon: probability of taking a uniform random action.
      rng: numpy random generator (for reproducibility).
      action_mask: optional bool array (n_actions,); True = legal action.
        If provided, both random and argmax pick only legal actions.

    Returns the chosen action index. Raises if no legal actions remain."""
    n = q_values.shape[0]
    legal = np.ones(n, dtype=bool) if action_mask is None else action_mask.astype(bool)
    if not legal.any():
        raise ValueError("no legal actions available")
    if rng.random() < epsilon:
        legal_idx = np.where(legal)[0]
        return int(rng.choice(legal_idx))
    # Greedy argmax restricted to legal actions
    q_masked = np.where(legal, q_values, -np.inf)
    return int(np.argmax(q_masked))
