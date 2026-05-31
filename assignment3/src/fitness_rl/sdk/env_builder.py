"""Build a :class:`WorldEnv` from config + an optional trained world model.

Factored out of the SDK so both training entry-points (REINFORCE, A2C)
and evaluation share the same env-construction logic.
"""

from __future__ import annotations

import numpy as np

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.shared.config import ConfigManager


def _identity_transition(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def build_env(
    config: ConfigManager,
    initial_state: np.ndarray,
    world_model: LSTMWorldModel | None,
) -> WorldEnv:
    """Construct a ``WorldEnv`` from config; LSTM is used as the dynamics if provided."""
    initial = initial_state.astype(np.float32, copy=True)
    if world_model is not None:
        transition_fn = world_model.as_transition_fn(
            window_size=int(config.get("world_model.window_size")),
            warmup_state=initial,
        )
    else:
        transition_fn = _identity_transition
    action_mask = (
        ActionMask(
            max_same_group=int(config.get("env.max_same_group_consecutive")),
            max_rest=int(config.get("env.max_rest_consecutive")),
        )
        if bool(config.get("env.action_masking_enabled"))
        else None
    )
    return WorldEnv(
        transition_fn=transition_fn,
        reward_fn=RewardFunction(
            gain_weight=float(config.get("env.reward_gain_weight")),
            overload_lambda=float(config.get("env.reward_overload_lambda")),
            imbalance_lambda=float(config.get("env.reward_imbalance_lambda")),
        ),
        initial_state=initial,
        episode_length=int(config.get("env.episode_length")),
        action_mask=action_mask,
    )
