"""Reward function: r_t = gain_t − λ_1·overload_t − λ_2·imbalance_t.

This is the contract between the fitness domain and the policy gradient
learner. The signal must reward useful training, penalise overload (rolling
high-volume risk), and penalise muscle-group imbalance (no policy collapse).
See ``docs/PRD_reward.md``.

**Layer 11 correction**: imbalance is now zeroed on the *action* the agent
took (``action == REST``), not the state's rest_indicator. The state-based
conditioning created an exploit where the policy could collect zero-imbalance
reward by steering toward states where the LSTM predicted rest_indicator high
regardless of the chosen action — see audit finding #10.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from fitness_rl.shared.types import Action

# Layout indices into the 16-dim state vector (see FeatureEngineer).
_VOL_IDX = 0
_MUSCLE_SLICE = slice(1, 6)
_N_MUSCLES = 5
_UNIFORM_ENTROPY = float(np.log(_N_MUSCLES))


class RewardFunction:
    """Stateful (maintains rolling volume window) — call ``reset()`` per episode."""

    def __init__(
        self,
        gain_weight: float = 1.0,
        overload_lambda: float = 0.2,
        imbalance_lambda: float = 0.3,
        rolling_window: int = 7,
    ):
        if any(x < 0 for x in (gain_weight, overload_lambda, imbalance_lambda)):
            raise ValueError("reward weights must be non-negative")
        if rolling_window < 1:
            raise ValueError("rolling_window must be >= 1")
        self._gain_weight = float(gain_weight)
        self._lambda_overload = float(overload_lambda)
        self._lambda_imbalance = float(imbalance_lambda)
        self._window: deque[float] = deque(maxlen=int(rolling_window))

    def reset(self) -> None:
        """Clear the rolling volume history. Call once at episode start."""
        self._window.clear()

    def compute(self, state: np.ndarray, action: int | None = None) -> float:
        """Compute reward from a 16-dim state vector + the action that produced it."""
        if state.shape[-1] != 16:
            raise ValueError(f"expected 16-dim state, got shape {state.shape}")
        volume = max(0.0, float(state[_VOL_IDX]))  # clamp negative LSTM outputs
        self._window.append(volume)
        gain = self._gain_weight * volume
        overload = self._lambda_overload * float(np.mean(self._window))
        is_rest_action = action is not None and int(action) == int(Action.REST)
        imbalance = 0.0 if is_rest_action else self._lambda_imbalance * self._imbalance(state)
        return gain - overload - imbalance

    def decompose(self, state: np.ndarray, action: int | None = None) -> dict[str, float]:
        """Return ``{gain, overload, imbalance, total}`` for diagnostics — no state mutation."""
        if state.shape[-1] != 16:
            raise ValueError(f"expected 16-dim state, got shape {state.shape}")
        volume = max(0.0, float(state[_VOL_IDX]))
        window_with_new = list(self._window) + [volume]
        gain = self._gain_weight * volume
        overload = self._lambda_overload * float(np.mean(window_with_new))
        is_rest_action = action is not None and int(action) == int(Action.REST)
        imbalance = 0.0 if is_rest_action else self._lambda_imbalance * self._imbalance(state)
        return {"gain": gain, "overload": overload, "imbalance": imbalance,
                "total": gain - overload - imbalance}

    @staticmethod
    def _imbalance(state: np.ndarray) -> float:
        """Imbalance ∈ [0, 1]: 0 when distribution is uniform, 1 when concentrated."""
        # Clamp negative values — the LSTM is unconstrained and can produce them,
        # but entropy on a probability distribution requires p_i ≥ 0.
        dist = np.maximum(0.0, np.asarray(state[_MUSCLE_SLICE], dtype=np.float64))
        total = dist.sum()
        if total <= 0:
            return 0.0
        p = dist / total
        eps = 1e-12
        entropy = float(-(p * np.log(p + eps)).sum())
        return max(0.0, min(1.0, 1.0 - entropy / _UNIFORM_ENTROPY))
