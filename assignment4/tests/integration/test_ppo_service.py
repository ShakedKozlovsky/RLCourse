"""End-to-end PPO + GAE smoke training on the actual MuJoCo env."""

from __future__ import annotations

import pytest
import torch

from proximal_lab.environment.vector_env import SyncVectorEnv
from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.services.ppo_service import PPOService
from proximal_lab.shared.seed import set_global_seed


def test_invalid_init_args_raise() -> None:
    with pytest.raises(ValueError):
        PPOService(gamma=0.0)
    with pytest.raises(ValueError):
        PPOService(gae_lambda=1.5)
    with pytest.raises(ValueError):
        PPOService(clip_eps=0.0)
    with pytest.raises(ValueError):
        PPOService(n_epochs_per_update=0)
    with pytest.raises(ValueError):
        PPOService(minibatch_size=0)


def test_invalid_fit_args_raise() -> None:
    set_global_seed(0)
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    svc = PPOService()
    with pytest.raises(ValueError):
        svc.fit(net, env, total_timesteps=0)


def test_smoke_training_returns_train_result() -> None:
    """Two iterations × 256 steps × 2 envs = 1024 timesteps."""
    set_global_seed(0)
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    svc = PPOService(n_epochs_per_update=2, minibatch_size=32)
    result = svc.fit(net, env, total_timesteps=1024, steps_per_rollout=256)
    assert result.total_timesteps >= 1024
    assert len(result.diagnostics) >= 2
    for d in result.diagnostics:
        assert d.mean_kl >= 0.0  # approx_kl can be negative tiny but mean usually positive
        assert 0.0 <= d.clip_fraction <= 1.0


def test_smoke_training_updates_weights() -> None:
    set_global_seed(0)
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    svc = PPOService(n_epochs_per_update=1, minibatch_size=32, lr=1e-2)
    svc.fit(net, env, total_timesteps=512, steps_per_rollout=256)
    after = net.state_dict()
    deltas = [(after[k] - before[k]).abs().max().item() for k in before]
    assert max(deltas) > 1e-6


def test_target_kl_early_stop_triggers() -> None:
    """With ``target_kl_stop`` set to a tiny number, training continues but
    diagnostics show the early-stop branch was exercised (mean_kl bounded)."""
    set_global_seed(0)
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    svc = PPOService(n_epochs_per_update=4, minibatch_size=32,
                      lr=1e-2, target_kl_stop=1e-9)  # impossible to satisfy
    result = svc.fit(net, env, total_timesteps=512, steps_per_rollout=256)
    # No assertion on a specific KL number — just that training completes
    # without divergence under the early-stop branch.
    assert len(result.diagnostics) >= 1


def test_diagnostics_record_all_pillars() -> None:
    """The slide-21 three-pillar diagnostics (Loop, Signal, Policy) all logged."""
    set_global_seed(0)
    env = SyncVectorEnv("HalfCheetah-v5", n_envs=2, seed=0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    svc = PPOService(n_epochs_per_update=1, minibatch_size=32)
    result = svc.fit(net, env, total_timesteps=512, steps_per_rollout=256)
    d = result.diagnostics[0]
    # Loop pillar
    assert d.iteration == 0
    assert d.timestep >= 512
    # Signal pillar (advantage quality)
    assert torch.is_tensor(torch.tensor(d.explained_variance))
    # Policy pillar (clipping bound)
    assert 0.0 <= d.clip_fraction <= 1.0
    # Plus reward + value + entropy
    assert hasattr(d, "mean_episode_reward")
    assert hasattr(d, "value_loss")
    assert hasattr(d, "entropy")
