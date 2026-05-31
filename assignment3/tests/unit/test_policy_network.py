"""PolicyNet — MLP forward contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as nn_f

from fitness_rl.model.policy_network import PolicyNet
from fitness_rl.shared.types import Action


def test_invalid_hidden_raises() -> None:
    with pytest.raises(ValueError):
        PolicyNet(hidden_size=0)


def test_forward_batch_shape() -> None:
    net = PolicyNet(hidden_size=32)
    x = torch.randn(4, net.state_dim)
    logits = net(x)
    assert logits.shape == (4, Action.n())


def test_forward_single_sample_shape() -> None:
    net = PolicyNet(hidden_size=32)
    x = torch.randn(net.state_dim)
    logits = net(x)
    assert logits.shape == (Action.n(),)


def test_forward_rejects_bad_shape() -> None:
    net = PolicyNet(hidden_size=32)
    with pytest.raises(ValueError):
        net(torch.randn(4, 10))
    with pytest.raises(ValueError):
        net(torch.randn(4, 7, 16))


def test_softmax_sums_to_one() -> None:
    net = PolicyNet(hidden_size=32)
    logits = net(torch.randn(8, net.state_dim))
    probs = nn_f.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)


def test_outputs_are_finite() -> None:
    torch.manual_seed(0)
    net = PolicyNet(hidden_size=16)
    logits = net(torch.randn(16, net.state_dim))
    assert torch.isfinite(logits).all()
