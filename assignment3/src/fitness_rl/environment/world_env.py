"""Gymnasium-style environment over a learned (or stub) transition function.

The environment is built generic over the transition function so this layer
can be fully tested before the LSTM (Layer 3) exists. Plug a real LSTM in
Layer 3, plug an identity function in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.action_space import ActionSpace
from fitness_rl.environment.reward import RewardFunction

# Callable signature: (current_state, action_int) -> next_state
TransitionFn = Callable[[np.ndarray, int], np.ndarray]


class WorldEnv:
    """RL environment whose dynamics are supplied as a callable transition function."""

    def __init__(
        self,
        transition_fn: TransitionFn,
        reward_fn: RewardFunction,
        initial_state: np.ndarray,
        episode_length: int,
        action_mask: ActionMask | None = None,
    ):
        if initial_state.shape != (16,):
            raise ValueError(f"initial_state must be shape (16,), got {initial_state.shape}")
        if episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        self._transition_fn = transition_fn
        self._reward_fn = reward_fn
        self._initial_state = initial_state.astype(np.float32, copy=True)
        self._episode_length = int(episode_length)
        self._action_mask = action_mask
        self.action_space = ActionSpace()
        self._state: np.ndarray = self._initial_state.copy()
        self._step_idx: int = 0
        self._recent_actions: list[int] = []

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset state and step counter; return ``(initial_state, info)``."""
        self._state = self._initial_state.copy()
        self._step_idx = 0
        self._recent_actions = []
        self._reward_fn.reset()
        return self._state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one day; return the Gymnasium 5-tuple."""
        if self._step_idx >= self._episode_length:
            raise RuntimeError("step() called on terminated env; reset() first")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        next_state = self._transition_fn(self._state, int(action)).astype(np.float32, copy=False)
        if next_state.shape != (16,):
            raise ValueError(f"transition_fn produced shape {next_state.shape}; expected (16,)")
        reward = self._reward_fn.compute(next_state, action=int(action))
        self._step_idx += 1
        self._recent_actions.append(int(action))
        terminated = self._step_idx >= self._episode_length
        info = {
            "step_idx": self._step_idx,
            "last_action": int(action),
            "recent_actions": list(self._recent_actions),
        }
        self._state = next_state
        return next_state.copy(), float(reward), terminated, False, info

    def get_mask(self) -> np.ndarray | None:
        """Return the current action mask (or None if masking is disabled)."""
        if self._action_mask is None:
            return None
        return self._action_mask.mask(self._recent_actions)
