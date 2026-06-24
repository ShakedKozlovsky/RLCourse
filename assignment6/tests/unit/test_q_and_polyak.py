"""Layer 4 — recurrent Q-net + Polyak 4-test math battery."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.soft_update import hard_copy, polyak_update

# ----- Polyak 4-test math battery -----

def _two_linears(seed_target: int, seed_source: int) -> tuple[nn.Linear, nn.Linear]:
    torch.manual_seed(seed_target)
    target = nn.Linear(4, 4)
    torch.manual_seed(seed_source)
    source = nn.Linear(4, 4)
    return target, source


def test_polyak_tau_zero_target_unchanged() -> None:
    target, source = _two_linears(1, 2)
    before = target.weight.clone()
    polyak_update(target.parameters(), source.parameters(), tau=0.0)
    torch.testing.assert_close(target.weight, before)


def test_polyak_tau_one_hard_copy() -> None:
    target, source = _two_linears(1, 2)
    polyak_update(target.parameters(), source.parameters(), tau=1.0)
    torch.testing.assert_close(target.weight, source.weight)


def test_polyak_tau_half_midpoint() -> None:
    target = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
        source.weight.fill_(1.0)
        source.bias.fill_(1.0)
    polyak_update(target.parameters(), source.parameters(), tau=0.5)
    assert torch.all(target.weight == 0.5)
    assert torch.all(target.bias == 0.5)


def test_polyak_repeated_converges() -> None:
    target = nn.Linear(2, 2)
    source = nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)
        source.weight.fill_(1.0)
        source.bias.fill_(1.0)
    for _ in range(200):
        polyak_update(target.parameters(), source.parameters(), tau=0.05)
    assert torch.allclose(target.weight, source.weight, atol=1e-3)


def test_polyak_invalid_tau_raises() -> None:
    target, source = _two_linears(0, 0)
    with pytest.raises(ValueError):
        polyak_update(target.parameters(), source.parameters(), tau=1.5)
    with pytest.raises(ValueError):
        polyak_update(target.parameters(), source.parameters(), tau=-0.1)


def test_hard_copy_matches_tau_one() -> None:
    target, source = _two_linears(1, 2)
    hard_copy(target, source)
    torch.testing.assert_close(target.weight, source.weight)


# ----- Recurrent Q-net -----

def test_q_per_agent_forward_shape() -> None:
    net = QPerAgent(obs_dim=18, n_actions=6, hidden_sizes=(64, 64),
                     gru_hidden_size=32)
    obs = torch.randn(4, 10, 18)   # batch=4, T=10
    q, h = net(obs)
    assert q.shape == (4, 10, 6)
    assert h.shape == (1, 4, 32)


def test_q_per_agent_handles_single_step() -> None:
    net = QPerAgent(obs_dim=18, n_actions=6)
    obs = torch.randn(4, 18)        # 2-D input (single timestep)
    q, h = net(obs)
    assert q.shape == (4, 1, 6)


def test_q_per_agent_hidden_propagates() -> None:
    """Calling forward twice with hidden chained should equal one long sequence."""
    net = QPerAgent(obs_dim=4, n_actions=3, gru_hidden_size=8)
    torch.manual_seed(0)
    obs1 = torch.randn(1, 3, 4)
    obs2 = torch.randn(1, 2, 4)
    # Chained
    q_a, h_a = net(obs1)
    q_b, h_b = net(obs2, hidden=h_a)
    # Single long sequence
    obs_long = torch.cat([obs1, obs2], dim=1)
    q_long, h_long = net(obs_long)
    torch.testing.assert_close(q_long[:, :3], q_a)
    torch.testing.assert_close(q_long[:, 3:], q_b)
    torch.testing.assert_close(h_long, h_b)


def test_q_per_agent_gradient_flows() -> None:
    net = QPerAgent(obs_dim=8, n_actions=5)
    obs = torch.randn(2, 5, 8)
    q, _ = net(obs)
    q.sum().backward()
    for p in net.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


def test_q_per_agent_init_hidden_shape() -> None:
    net = QPerAgent(obs_dim=4, n_actions=3, gru_hidden_size=16)
    h = net.init_hidden(batch_size=7)
    assert h.shape == (1, 7, 16)
    assert torch.all(h == 0.0)
