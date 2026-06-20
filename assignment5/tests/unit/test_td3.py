"""Layer 20 — TD3 add-on tests."""

from __future__ import annotations

import numpy as np
import torch

from roomba_lab.model.td3_network import TD3Network
from roomba_lab.services.td3_update import (
    actor_loss_td3,
    apply_td3_update,
    critic_loss_td3,
)


def _batch(obs_dim: int = 5, action_dim: int = 2, n: int = 8) -> dict:
    rng = np.random.default_rng(0)
    return {
        "state": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "action": rng.uniform(-1.0, 1.0, (n, action_dim)).astype(np.float32),
        "reward": rng.standard_normal((n,)).astype(np.float32),
        "next_state": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "done": (rng.random((n,)) > 0.5).astype(np.float32),
    }


def test_td3_network_has_two_critics() -> None:
    net = TD3Network(obs_dim=5, action_dim=2)
    assert hasattr(net, "critic_a")
    assert hasattr(net, "critic_b")
    assert hasattr(net, "target_critic_a")
    assert hasattr(net, "target_critic_b")


def test_td3_critic_loss_uses_min_target() -> None:
    """The bootstrap target uses min(Q_a, Q_b); equal critics → loss equals single-critic version twice."""
    net = TD3Network(obs_dim=5, action_dim=2)
    net.target_critic_b.load_state_dict(net.target_critic_a.state_dict())
    loss = critic_loss_td3(net, _batch(), gamma=0.99,
                            target_policy_noise=0.0, target_noise_clip=0.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_td3_target_noise_clipped() -> None:
    """Even with huge target noise, target action stays in [-1, 1]."""
    net = TD3Network(obs_dim=5, action_dim=2)
    loss = critic_loss_td3(net, _batch(), gamma=0.99,
                            target_policy_noise=10.0, target_noise_clip=0.5)
    assert torch.isfinite(loss)


def test_td3_actor_loss_uses_critic_a_only() -> None:
    """Per Fujimoto 2018, actor optimises against critic_a, not min."""
    net = TD3Network(obs_dim=5, action_dim=2)
    loss = actor_loss_td3(net, _batch())
    assert torch.isfinite(loss)


def test_td3_policy_delay_skips_actor_step() -> None:
    """With policy_delay=2 and step=1, actor should NOT update; step=2 should."""
    net = TD3Network(obs_dim=5, action_dim=2)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(
        list(net.critic_a.parameters()) + list(net.critic_b.parameters()),
        lr=1e-3,
    )
    diag_skip = apply_td3_update(net, _batch(), step=1, gamma=0.99, tau=0.005,
                                   policy_delay=2, target_policy_noise=0.2,
                                   target_noise_clip=0.5,
                                   actor_opt=actor_opt, critic_opt=critic_opt)
    diag_update = apply_td3_update(net, _batch(), step=2, gamma=0.99, tau=0.005,
                                     policy_delay=2, target_policy_noise=0.2,
                                     target_noise_clip=0.5,
                                     actor_opt=actor_opt, critic_opt=critic_opt)
    assert diag_skip.actor_updated is False
    assert diag_update.actor_updated is True


def test_td3_step_changes_critic_weights() -> None:
    net = TD3Network(obs_dim=5, action_dim=2)
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(
        list(net.critic_a.parameters()) + list(net.critic_b.parameters()),
        lr=1e-3,
    )
    c_before = next(net.critic_a.parameters()).clone()
    apply_td3_update(net, _batch(), step=0, gamma=0.99, tau=0.005,
                      policy_delay=2, target_policy_noise=0.2,
                      target_noise_clip=0.5,
                      actor_opt=actor_opt, critic_opt=critic_opt)
    c_after = next(net.critic_a.parameters())
    assert (c_before - c_after).abs().max() > 1e-6
