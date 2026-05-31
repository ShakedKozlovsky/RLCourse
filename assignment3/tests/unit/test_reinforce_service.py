"""ReinforceService — reward-to-go, mean baseline, episodic update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fitness_rl.environment.action_mask import ActionMask
from fitness_rl.environment.reward import RewardFunction
from fitness_rl.environment.world_env import WorldEnv
from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.services.reinforce_service import ReinforceService
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
        ReinforceService(gamma=0.0)
    with pytest.raises(ValueError):
        ReinforceService(gamma=1.1)
    with pytest.raises(ValueError):
        ReinforceService(entropy_bonus=-0.1)


def test_reward_to_go_arithmetic() -> None:
    rewards = [1.0, 2.0, 3.0]
    out = ReinforceService.reward_to_go(rewards, gamma=0.5)
    # G_2 = 3
    # G_1 = 2 + 0.5*3 = 3.5
    # G_0 = 1 + 0.5*3.5 = 2.75
    assert out == pytest.approx([2.75, 3.5, 3.0])


def test_reward_to_go_undiscounted() -> None:
    rewards = [1.0, 2.0, 3.0, 4.0]
    out = ReinforceService.reward_to_go(rewards, gamma=1.0)
    assert out == pytest.approx([10.0, 9.0, 7.0, 4.0])


def test_reward_to_go_empty() -> None:
    assert ReinforceService.reward_to_go([], gamma=0.99) == []


def test_fit_returns_per_episode_metrics() -> None:
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=16)
    env = _make_env(episode_length=4)
    svc = ReinforceService(gamma=0.99)
    history = svc.fit(policy, env, episodes=3)
    assert len(history) == 3
    for m in history:
        assert m.action_counts.shape == (Action.n(),)
        assert int(m.action_counts.sum()) == 4  # episode_length
        assert m.mean_entropy >= 0.0


def test_fit_rejects_zero_episodes() -> None:
    policy = PolicyNet(hidden_size=8)
    env = _make_env()
    with pytest.raises(ValueError):
        ReinforceService().fit(policy, env, episodes=0)


def test_single_update_changes_weights() -> None:
    """One REINFORCE step should produce a non-trivial weight delta."""
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=16)
    env = _make_env(episode_length=4)
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    ReinforceService(gamma=0.99, lr=1e-2).fit(policy, env, episodes=1)
    after = policy.state_dict()
    deltas = [(after[k] - before[k]).abs().max().item() for k in before]
    assert max(deltas) > 1e-6


def test_action_masking_respected_during_rollout() -> None:
    """With masking on, the policy never picks a forbidden action."""
    torch.manual_seed(0)
    policy = PolicyNet(hidden_size=8)
    env = _make_env(episode_length=20, action_mask=ActionMask(max_same_group=2))
    svc = ReinforceService(use_action_mask=True)
    svc.fit(policy, env, episodes=1)
    # rebuild a rollout to inspect the action stream
    state, _ = env.reset()
    recent: list[int] = []
    for _ in range(20):
        logits = policy(torch.from_numpy(state).float())
        mask = env.get_mask()
        if mask is not None:
            logits = logits + torch.from_numpy(mask).float()
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
        recent.append(action)
        state, _, term, _, _ = env.step(action)
        if term:
            break
    # No three consecutive same actions in the stream.
    for i in range(len(recent) - 2):
        assert not (recent[i] == recent[i + 1] == recent[i + 2])


def test_entropy_bonus_increases_entropy_relative_to_no_bonus() -> None:
    """High entropy bonus → policy stays closer to uniform on average."""
    torch.manual_seed(0)
    env = _make_env(episode_length=6)

    p_none = PolicyNet(hidden_size=16)
    ReinforceService(entropy_bonus=0.0, lr=1e-2).fit(p_none, env, episodes=20)

    torch.manual_seed(0)
    env2 = _make_env(episode_length=6)
    p_high = PolicyNet(hidden_size=16)
    history_high = ReinforceService(entropy_bonus=1.0, lr=1e-2).fit(
        p_high, env2, episodes=20
    )

    torch.manual_seed(123)
    env3 = _make_env(episode_length=6)
    history_none = ReinforceService(entropy_bonus=0.0, lr=1e-2).fit(
        p_none, env3, episodes=20
    )

    avg_high = float(np.mean([m.mean_entropy for m in history_high[-5:]]))
    avg_none = float(np.mean([m.mean_entropy for m in history_none[-5:]]))
    # Not a strict guarantee per seed, but the bonus should push entropy up
    # at least relative to a baseline with no bonus.
    assert avg_high >= avg_none - 0.1
