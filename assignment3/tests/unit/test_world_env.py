"""WorldEnv — Gymnasium-style reset/step contract with stub transition fn."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.shared.types import Action


def _identity(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def _make_env(episode_length: int = 3,
              transition_fn=_identity,
              action_mask: ActionMask | None = None) -> WorldEnv:
    initial = np.zeros(16, dtype=np.float32)
    initial[0] = 0.5
    initial[1:6] = 0.2
    return WorldEnv(
        transition_fn=transition_fn,
        reward_fn=RewardFunction(),
        initial_state=initial,
        episode_length=episode_length,
        action_mask=action_mask,
    )


def test_invalid_initial_shape_raises() -> None:
    with pytest.raises(ValueError):
        WorldEnv(_identity, RewardFunction(), np.zeros(5, dtype=np.float32), 3)


def test_invalid_episode_length_raises() -> None:
    with pytest.raises(ValueError):
        WorldEnv(_identity, RewardFunction(), np.zeros(16, dtype=np.float32), 0)


def test_reset_returns_initial_state() -> None:
    env = _make_env()
    state, info = env.reset()
    assert state.shape == (16,)
    assert info == {}
    assert state[0] == 0.5


def test_step_returns_five_tuple() -> None:
    env = _make_env()
    env.reset()
    out = env.step(int(Action.PUSH))
    assert len(out) == 5
    next_state, reward, terminated, truncated, info = out
    assert next_state.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert info["step_idx"] == 1


def test_terminated_after_episode_length() -> None:
    env = _make_env(episode_length=2)
    env.reset()
    _, _, term1, _, _ = env.step(int(Action.PUSH))
    _, _, term2, _, _ = env.step(int(Action.PULL))
    assert term1 is False
    assert term2 is True


def test_step_after_termination_raises() -> None:
    env = _make_env(episode_length=1)
    env.reset()
    env.step(int(Action.PUSH))
    with pytest.raises(RuntimeError):
        env.step(int(Action.PUSH))


def test_invalid_action_raises() -> None:
    env = _make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_transition_shape_validated() -> None:
    def bad_transition(state: np.ndarray, action: int) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)
    env = _make_env(transition_fn=bad_transition)
    env.reset()
    with pytest.raises(ValueError):
        env.step(int(Action.PUSH))


def test_get_mask_none_without_mask() -> None:
    env = _make_env()
    env.reset()
    assert env.get_mask() is None


def test_get_mask_reflects_recent_actions() -> None:
    env = _make_env(episode_length=5, action_mask=ActionMask(max_same_group=2))
    env.reset()
    env.step(int(Action.PUSH))
    env.step(int(Action.PUSH))
    mask = env.get_mask()
    assert mask is not None
    assert mask[Action.PUSH] == -np.inf


def test_reset_clears_history() -> None:
    env = _make_env(episode_length=5, action_mask=ActionMask(max_same_group=2))
    env.reset()
    env.step(int(Action.PUSH))
    env.step(int(Action.PUSH))
    env.reset()
    mask = env.get_mask()
    assert mask is not None
    assert np.all(mask == 0.0)


def test_info_contains_recent_actions() -> None:
    env = _make_env(episode_length=3)
    env.reset()
    env.step(int(Action.PUSH))
    _, _, _, _, info = env.step(int(Action.PULL))
    assert info["recent_actions"] == [int(Action.PUSH), int(Action.PULL)]
    assert info["last_action"] == int(Action.PULL)


def test_state_advances_with_non_identity_transition() -> None:
    def bump(state: np.ndarray, action: int) -> np.ndarray:
        out = state.copy()
        out[0] = min(1.0, state[0] + 0.1)
        return out
    env = _make_env(episode_length=2, transition_fn=bump)
    s0, _ = env.reset()
    s1, _, _, _, _ = env.step(int(Action.PUSH))
    assert s1[0] > s0[0]
