"""Pre-softmax action masking — expert guardrails over the policy.

The mask is a (n_actions,) array of zeros (allowed) and ``-inf`` (forbidden).
Adding the mask to the policy's logits *before* softmax zeroes the probability
of forbidden actions while preserving the gradient through allowed actions.
See ``docs/PRD_action_masking.md``.
"""

from __future__ import annotations

import numpy as np

from fitness_rl.shared.types import Action


class ActionMask:
    """Forbid 3 consecutive same-group days and 3 consecutive REST days."""

    def __init__(self, max_same_group: int = 2, max_rest: int = 2):
        if max_same_group < 1 or max_rest < 1:
            raise ValueError("max_* must be >= 1")
        self._max_same = int(max_same_group)
        self._max_rest = int(max_rest)
        self._n = Action.n()

    def mask(self, recent_actions: list[int]) -> np.ndarray:
        """Return a (n_actions,) float array of 0s and -inf values."""
        out = np.zeros(self._n, dtype=np.float32)
        if not recent_actions:
            return out
        # Same-group rule: if last max_same actions were all action X, forbid X.
        if len(recent_actions) >= self._max_same:
            last_n = recent_actions[-self._max_same :]
            if len(set(last_n)) == 1:
                out[last_n[0]] = -np.inf
        # Extra rest rule: if last max_rest actions were all REST, forbid REST.
        if len(recent_actions) >= self._max_rest:
            last_n = recent_actions[-self._max_rest :]
            if all(a == Action.REST for a in last_n):
                out[Action.REST] = -np.inf
        return out
