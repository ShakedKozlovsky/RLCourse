"""DuelingDQN — shapes, Dueling identity, vanilla fallback, sync helpers."""

from __future__ import annotations

import pytest
import torch

from dqn_trader.model.dueling_dqn import DuelingDQN, hard_update, soft_update


def test_forward_shape_dueling() -> None:
    net = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    out = net(torch.zeros(4, 30, 10))
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_forward_shape_vanilla() -> None:
    net = DuelingDQN(window_size=30, n_features=10, n_actions=3, dueling=False)
    out = net(torch.zeros(4, 30, 10))
    assert out.shape == (4, 3)


def test_dueling_identity_holds() -> None:
    """For Dueling: Q[a] − Q.mean = A[a] − A.mean. Pure structural property."""
    torch.manual_seed(0)
    net = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    x = torch.randn(2, 30, 10)
    q = net(x)
    # The Dueling aggregation enforces Q.mean(dim=1) == V, so q.mean is V repeated.
    centered = q - q.mean(dim=1, keepdim=True)
    # Recover A by passing the trunk + advantage head manually.
    feats = net.trunk(x.permute(0, 2, 1))
    h = net.shared_fc(feats)
    a = net.advantage_head(h)
    a_centered = a - a.mean(dim=1, keepdim=True)
    torch.testing.assert_close(centered, a_centered, atol=1e-6, rtol=1e-6)


def test_invalid_shape_raises() -> None:
    net = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    with pytest.raises(ValueError):
        net(torch.zeros(30, 10))  # missing batch dimension


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        DuelingDQN(window_size=1, n_features=10, n_actions=3)
    with pytest.raises(ValueError):
        DuelingDQN(window_size=30, n_features=10, n_actions=1)


def test_hard_update_copies_weights() -> None:
    src = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    tgt = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    hard_update(tgt, src)
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        torch.testing.assert_close(sp, tp)


def test_soft_update_polyak_blend() -> None:
    src = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    tgt = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    soft_update(tgt, src, tau=1.0)  # full overwrite ≡ hard update
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        torch.testing.assert_close(sp, tp)


def test_soft_update_tau_bounds() -> None:
    src = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    tgt = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    with pytest.raises(ValueError):
        soft_update(tgt, src, tau=0.0)
    with pytest.raises(ValueError):
        soft_update(tgt, src, tau=1.5)


def test_backward_pass_runs() -> None:
    """Repeated forward/backward/step decreases loss on a fixed synthetic target."""
    torch.manual_seed(0)
    net = DuelingDQN(window_size=30, n_features=10, n_actions=3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(8, 30, 10)
    target = torch.randn(8, 3)
    loss0 = ((net(x) - target) ** 2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    loss_final = ((net(x) - target) ** 2).mean().item()
    assert loss_final < loss0
