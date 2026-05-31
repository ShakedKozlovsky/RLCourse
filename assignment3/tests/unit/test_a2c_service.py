"""A2CService — TD-error advantage, two-optimizer update, masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.services.a2c_service import A2CService
from fitness_rl.shared.types import Action


def _identity(state: np.ndarray, action: int) -> np.ndarray:
    return state.copy()


def _make_env(episode_length: int = 5,
              action_mask: ActionMask | None = None) -> WorldEnv:
    initial = np.zeros(16, dtype=np.float32)
    initial[0] = 0.5
    initial[1:6] = 0.2
    return WorldEnv(
        transition_fn=_identity,
        reward_fn=RewardFunction(),
        initial_state=initial,
        episode_length=episode_length,
        action_mask=action_mask,
    )


def test_invalid_init_args_raise() -> None:
    with pytest.raises(ValueError):
        A2CService(gamma=0.0)
    with pytest.raises(ValueError):
        A2CService(gamma=1.5)
    with pytest.raises(ValueError):
        A2CService(entropy_bonus=-0.5)


def test_td_error_positive_when_next_better_and_reward_positive() -> None:
    delta = A2CService.td_error(reward=1.0, value=0.2, next_value=0.5,
                                gamma=0.99, terminated=False)
    # δ = 1.0 + 0.99 * 0.5 - 0.2 = 1.295
    assert delta == pytest.approx(1.295)
    assert delta > 0


def test_td_error_negative_when_value_overestimates() -> None:
    delta = A2CService.td_error(reward=-0.5, value=1.0, next_value=0.2,
                                gamma=0.99, terminated=False)
    # δ = -0.5 + 0.99*0.2 - 1.0 = -1.302
    assert delta < 0


def test_td_error_zeroes_bootstrap_on_terminal() -> None:
    delta = A2CService.td_error(reward=0.3, value=0.1, next_value=99.0,
                                gamma=0.99, terminated=True)
    # next_value ignored: δ = 0.3 - 0.1 = 0.2
    assert delta == pytest.approx(0.2)


def test_fit_returns_per_episode_metrics() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=16)
    env = _make_env(episode_length=4)
    history = A2CService(gamma=0.99).fit(net, env, episodes=3)
    assert len(history) == 3
    for m in history:
        assert m.action_counts.shape == (Action.n(),)
        assert int(m.action_counts.sum()) == 4
        assert m.mean_entropy >= 0.0


def test_fit_rejects_zero_episodes() -> None:
    net = ActorCriticNet(hidden_size=8)
    env = _make_env()
    with pytest.raises(ValueError):
        A2CService().fit(net, env, episodes=0)


def test_single_episode_changes_weights_in_both_heads() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=16)
    env = _make_env(episode_length=4)
    actor_before = {k: v.detach().clone() for k, v in net.actor_head.state_dict().items()}
    critic_before = {k: v.detach().clone() for k, v in net.critic_head.state_dict().items()}
    A2CService(gamma=0.99, actor_lr=1e-2, critic_lr=1e-2).fit(net, env, episodes=1)
    actor_after = net.actor_head.state_dict()
    critic_after = net.critic_head.state_dict()
    actor_delta = max((actor_after[k] - actor_before[k]).abs().max().item()
                      for k in actor_before)
    critic_delta = max((critic_after[k] - critic_before[k]).abs().max().item()
                       for k in critic_before)
    assert actor_delta > 1e-6
    assert critic_delta > 1e-6


def test_critic_value_drifts_with_training() -> None:
    """Identity-transition env has near-fixed rewards → critic V(s) should
    move away from its random-init value as training progresses."""
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=32)
    env = _make_env(episode_length=10)
    initial_state = torch.from_numpy(env.reset()[0]).float()
    v0_before = net(initial_state)[1].item()
    A2CService(gamma=0.99, actor_lr=1e-3, critic_lr=1e-2).fit(net, env, episodes=30)
    v0_after = net(initial_state)[1].item()
    assert abs(v0_after - v0_before) > 1e-3


def test_action_masking_respected_during_rollout() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=8)
    env = _make_env(episode_length=20, action_mask=ActionMask(max_same_group=2))
    A2CService(use_action_mask=True).fit(net, env, episodes=1)
    # Replay one greedy-sampled rollout with masking to verify the policy
    # respects the constraint.
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
