"""Layer 4 — Actor + Critic + ActorCriticNet contract tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from roomba_lab.model.actor import Actor
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.model.critic import Critic


def test_actor_output_in_unit_box() -> None:
    a = Actor(obs_dim=29, action_dim=2)
    obs = torch.randn(8, 29)
    out = a(obs)
    assert out.shape == (8, 2)
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_critic_output_scalar_per_batch() -> None:
    c = Critic(obs_dim=29, action_dim=2)
    obs = torch.randn(8, 29)
    act = torch.randn(8, 2)
    q = c(obs, act)
    assert q.shape == (8,)


def test_actor_gradient_flows() -> None:
    a = Actor(obs_dim=5, action_dim=2)
    obs = torch.randn(3, 5)
    out = a(obs).sum()
    out.backward()
    for p in a.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_critic_gradient_flows() -> None:
    c = Critic(obs_dim=5, action_dim=2)
    obs = torch.randn(3, 5)
    act = torch.randn(3, 2)
    q = c(obs, act).sum()
    q.backward()
    for p in c.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_actor_critic_net_targets_frozen() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    for p in net.target_actor.parameters():
        assert p.requires_grad is False
    for p in net.target_critic.parameters():
        assert p.requires_grad is False


def test_actor_critic_save_load_roundtrip() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    obs = torch.randn(3, 5)
    out_before = net.actor(obs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ac.pt"
        net.save(path)
        loaded = ActorCriticNet.load(path, obs_dim=5, action_dim=2)
        out_after = loaded.actor(obs)
    torch.testing.assert_close(out_after, out_before)


def test_act_disables_grad() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    obs = torch.randn(3, 5)
    out = net.act(obs)
    assert out.requires_grad is False
