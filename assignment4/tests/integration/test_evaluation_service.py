"""EvaluationService — n-episode rollout aggregation."""

from __future__ import annotations

import pytest

from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.services.evaluation_service import (
    EvaluationResult,
    EvaluationService,
)
from proximal_lab.shared.seed import set_global_seed


def test_invalid_n_episodes_raises() -> None:
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    with pytest.raises(ValueError):
        EvaluationService().rollout(net, "HalfCheetah-v5", n_episodes=0)


def test_rollout_returns_eval_result() -> None:
    set_global_seed(0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    result = EvaluationService().rollout(net, "HalfCheetah-v5", n_episodes=2, max_steps=50)
    assert isinstance(result, EvaluationResult)
    assert result.n_episodes == 2
    assert len(result.per_episode_rewards) == 2
    assert isinstance(result.mean_reward, float)


def test_deterministic_rollout_same_seed_reproducible() -> None:
    set_global_seed(0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    a = EvaluationService(deterministic=True).rollout(
        net, "HalfCheetah-v5", n_episodes=1, seed=42, max_steps=30)
    b = EvaluationService(deterministic=True).rollout(
        net, "HalfCheetah-v5", n_episodes=1, seed=42, max_steps=30)
    # Same seed, same network, deterministic: should produce equal rollouts.
    assert a.per_episode_rewards[0] == pytest.approx(b.per_episode_rewards[0], abs=1e-5)


def test_walker2d_evaluation() -> None:
    set_global_seed(0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    result = EvaluationService().rollout(net, "Walker2d-v5", n_episodes=1, max_steps=20)
    assert result.n_episodes == 1
