"""ActorCriticNet — act + evaluate + actor/critic param partition + save/load."""

from __future__ import annotations

from pathlib import Path

import torch

from proximal_lab.model.actor_critic_network import ActorCriticNet


def test_act_shapes() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(obs_dim=17, action_dim=6)
    obs = torch.randn(4, 17)
    action, log_prob, value = net.act(obs)
    assert action.shape == (4, 6)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)


def test_act_deterministic_returns_distribution_mean() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(obs_dim=4, action_dim=2)
    obs = torch.randn(3, 4)
    dist = net.actor(obs)
    action_det, _, _ = net.act(obs, deterministic=True)
    assert torch.allclose(action_det, dist.mean)


def test_evaluate_returns_log_prob_entropy_value() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(obs_dim=4, action_dim=2)
    obs = torch.randn(5, 4)
    actions = torch.randn(5, 2)
    log_prob, entropy, value = net.evaluate(obs, actions)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5,)


def test_actor_critic_params_disjoint() -> None:
    """Per ADR-002: separate networks → no shared parameters."""
    net = ActorCriticNet(obs_dim=4, action_dim=2)
    actor_ids = {id(p) for p in net.actor_params()}
    critic_ids = {id(p) for p in net.critic_params()}
    assert actor_ids.isdisjoint(critic_ids)


def test_save_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original = ActorCriticNet(obs_dim=17, action_dim=6)
    obs = torch.randn(3, 17)
    with torch.no_grad():
        before_action, before_log_prob, before_value = original.act(obs, deterministic=True)
    ckpt = tmp_path / "net.pt"
    original.save(ckpt)
    restored = ActorCriticNet.load(ckpt)
    with torch.no_grad():
        after_action, _, after_value = restored.act(obs, deterministic=True)
    assert torch.allclose(before_action, after_action, atol=1e-6)
    assert torch.allclose(before_value, after_value, atol=1e-6)
