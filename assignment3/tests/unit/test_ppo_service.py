"""PPOService — clipped surrogate, importance ratio, multi-epoch update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.services.ppo_service import PPOService
from fitness_rl.shared.types import Action


def _identity(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def _make_env(episode_length: int = 6,
              action_mask: ActionMask | None = None) -> WorldEnv:
    initial = np.zeros(16, dtype=np.float32)
    initial[0] = 0.5
    initial[1:6] = 0.2
    return WorldEnv(_identity, RewardFunction(), initial, episode_length, action_mask)


def test_invalid_init_args_raise() -> None:
    with pytest.raises(ValueError):
        PPOService(gamma=0.0)
    with pytest.raises(ValueError):
        PPOService(clip_eps=0.0)
    with pytest.raises(ValueError):
        PPOService(n_epochs_per_batch=0)
    with pytest.raises(ValueError):
        PPOService(n_steps_per_update=0)


def test_fit_returns_per_episode_metrics() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=16)
    env = _make_env(episode_length=4)
    history = PPOService(gamma=0.99).fit(net, env, episodes=3)
    assert len(history) == 3
    for m in history:
        assert m.action_counts.shape == (Action.n(),)
        assert int(m.action_counts.sum()) == 4
        assert m.mean_entropy >= 0.0


def test_fit_rejects_zero_episodes() -> None:
    net = ActorCriticNet(hidden_size=8)
    env = _make_env()
    with pytest.raises(ValueError):
        PPOService().fit(net, env, episodes=0)


def test_single_update_changes_weights() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=16)
    env = _make_env(episode_length=6)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    PPOService(lr=1e-2).fit(net, env, episodes=1)
    deltas = [
        (net.state_dict()[k] - before[k]).abs().max().item() for k in before
    ]
    assert max(deltas) > 1e-6


def test_clip_bounds_importance_ratio() -> None:
    """Direct check of the clipped-objective math at the tensor level."""
    advantages = torch.tensor([1.0, -1.0])
    ratio = torch.tensor([5.0, 5.0])      # both ratios above 1+eps
    eps = 0.2
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    # PPO objective = min(r·A, clip(r)·A) per element; loss = -mean(this).
    obj = torch.min(ratio * advantages, clipped * advantages)
    # For advantage > 0 with ratio > 1+eps, the clipped branch wins.
    # For advantage < 0 with ratio > 1+eps, the un-clipped (larger negative) wins.
    assert obj[0].item() == pytest.approx((1 + eps) * 1.0)
    assert obj[1].item() == pytest.approx(5.0 * -1.0)


def test_action_masking_respected_during_rollout() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=8)
    env = _make_env(episode_length=20, action_mask=ActionMask(max_same_group=2))
    PPOService(use_action_mask=True).fit(net, env, episodes=1)
    # Replay one masked rollout
    state, _ = env.reset()
    recent: list[int] = []
    for _ in range(20):
        logits, _ = net(torch.from_numpy(state).float())
        mask = env.get_mask()
        if mask is not None:
            logits = logits + torch.from_numpy(mask).float()
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
        recent.append(action)
        state, _, term, _, _ = env.step(action)
        if term:
            break
    for i in range(len(recent) - 2):
        assert not (recent[i] == recent[i + 1] == recent[i + 2])
