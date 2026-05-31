"""EvaluationService — greedy rollout, action distribution, collapse detection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.services.evaluation_service import (
    EvaluationResult,
    EvaluationService,
    actor_logits,
)
from fitness_rl.shared.types import Action


def _identity(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def _make_env(episode_length: int = 5,
              action_mask: ActionMask | None = None) -> WorldEnv:
    initial = np.zeros(16, dtype=np.float32)
    initial[0] = 0.5
    initial[1:6] = 0.2
    return WorldEnv(_identity, RewardFunction(), initial, episode_length, action_mask)


def test_rollout_basic_shapes() -> None:
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=8)
    env = _make_env(episode_length=4)
    result = EvaluationService().rollout(actor_logits(policy), env)
    assert isinstance(result, EvaluationResult)
    assert len(result.action_sequence) == 4
    assert len(result.rewards) == 4
    assert len(result.states) == 5  # initial + 4 steps
    assert int(result.action_counts.sum()) == 4


def test_rollout_is_deterministic() -> None:
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=8)
    env_a = _make_env(episode_length=6)
    env_b = _make_env(episode_length=6)
    r1 = EvaluationService().rollout(actor_logits(policy), env_a)
    r2 = EvaluationService().rollout(actor_logits(policy), env_b)
    assert r1.action_sequence == r2.action_sequence


def test_rollout_with_actor_critic_uses_logits_only() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=8)
    env = _make_env(episode_length=4)
    result = EvaluationService().rollout(actor_logits(net), env)
    assert int(result.action_counts.sum()) == 4


def test_action_distribution_normalised() -> None:
    counts = np.array([2, 0, 1, 1, 0], dtype=np.int64)
    result = EvaluationResult(
        total_reward=0.0, action_counts=counts,
        action_sequence=[0, 0, 2, 3], rewards=[0.0] * 4, states=[],
    )
    dist = EvaluationService.action_distribution(result)
    assert dist.sum() == pytest.approx(1.0)
    assert dist[0] == pytest.approx(0.5)


def test_action_distribution_empty_returns_zeros() -> None:
    counts = np.zeros(Action.n(), dtype=np.int64)
    result = EvaluationResult(0.0, counts, [], [], [])
    dist = EvaluationService.action_distribution(result)
    assert dist.sum() == 0.0


def test_collapsed_detects_concentration() -> None:
    counts = np.array([10, 0, 0, 0, 0], dtype=np.int64)
    result = EvaluationResult(0.0, counts, [0] * 10, [0.0] * 10, [])
    assert EvaluationService.collapsed(result, threshold=0.8) is True


def test_not_collapsed_when_diverse() -> None:
    counts = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    result = EvaluationResult(0.0, counts, [], [], [])
    assert EvaluationService.collapsed(result, threshold=0.8) is False


def test_rollout_respects_action_mask() -> None:
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=8)
    env = _make_env(episode_length=20, action_mask=ActionMask(max_same_group=2))
    result = EvaluationService(respect_action_mask=True).rollout(actor_logits(policy), env)
    for i in range(len(result.action_sequence) - 2):
        a, b, c = result.action_sequence[i : i + 3]
        assert not (a == b == c), f"3-in-a-row at {i}: {a}/{b}/{c}"


def test_rollout_can_ignore_mask() -> None:
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=8)
    env = _make_env(episode_length=4, action_mask=ActionMask(max_same_group=2))
    # Should not crash even with respect_action_mask=False
    result = EvaluationService(respect_action_mask=False).rollout(actor_logits(policy), env)
    assert len(result.action_sequence) == 4
