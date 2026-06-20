"""Layer 7 — DDPG update step gradient-flow + math tests."""

from __future__ import annotations

import numpy as np
import torch

from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.services.ddpg_update import actor_loss, apply_update, critic_loss


def _batch(obs_dim: int, action_dim: int, n: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "state": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "action": rng.uniform(-1.0, 1.0, (n, action_dim)).astype(np.float32),
        "reward": rng.standard_normal((n,)).astype(np.float32),
        "next_state": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "done": (rng.random((n,)) > 0.5).astype(np.float32),
    }


def test_critic_loss_is_finite() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    loss = critic_loss(net, _batch(5, 2), gamma=0.99)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_critic_loss_backward_flows_only_to_critic() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    loss = critic_loss(net, _batch(5, 2), gamma=0.99)
    loss.backward()
    assert any(p.grad is not None for p in net.critic.parameters())
    for p in net.actor.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
    for p in net.target_critic.parameters():
        assert p.grad is None


def test_actor_loss_backward_flows_to_actor() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    a_loss = actor_loss(net, _batch(5, 2))
    a_loss.backward()
    assert any(p.grad is not None for p in net.actor.parameters())


def test_one_step_changes_weights() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(net.critic.parameters(), lr=1e-3)
    a_before = next(net.actor.parameters()).clone()
    c_before = next(net.critic.parameters()).clone()
    apply_update(net, _batch(5, 2), gamma=0.99, tau=0.005,
                 actor_opt=actor_opt, critic_opt=critic_opt)
    a_after = next(net.actor.parameters())
    c_after = next(net.critic.parameters())
    assert (a_before - a_after).abs().max() > 1e-6
    assert (c_before - c_after).abs().max() > 1e-6


def test_apply_update_returns_diagnostic() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(net.critic.parameters(), lr=1e-3)
    diag = apply_update(net, _batch(5, 2), gamma=0.99, tau=0.005,
                        actor_opt=actor_opt, critic_opt=critic_opt)
    assert isinstance(diag.critic_loss, float)
    assert isinstance(diag.actor_loss, float)
    assert isinstance(diag.mean_q, float)
    assert isinstance(diag.target_drift, float)
    assert diag.target_drift > 0.0


def test_polyak_step_with_tau_zero_freezes_target() -> None:
    net = ActorCriticNet(obs_dim=5, action_dim=2)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(net.critic.parameters(), lr=1e-3)
    target_before = next(net.target_critic.parameters()).clone()
    apply_update(net, _batch(5, 2), gamma=0.99, tau=0.0,
                 actor_opt=actor_opt, critic_opt=critic_opt)
    target_after = next(net.target_critic.parameters())
    torch.testing.assert_close(target_before, target_after)
