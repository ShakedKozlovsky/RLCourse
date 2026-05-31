"""ActorCriticNet — shared trunk + actor/critic head shapes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as nn_f

from fitness_rl.model.actor_critic_network import ActorCriticNet
from fitness_rl.shared.types import Action


def test_invalid_hidden_raises() -> None:
    with pytest.raises(ValueError):
        ActorCriticNet(hidden_size=0)


def test_forward_batch_shapes() -> None:
    net = ActorCriticNet(hidden_size=32)
    x = torch.randn(4, net.state_dim)
    logits, value = net(x)
    assert logits.shape == (4, Action.n())
    assert value.shape == (4,)


def test_forward_single_sample_shapes() -> None:
    net = ActorCriticNet(hidden_size=32)
    x = torch.randn(net.state_dim)
    logits, value = net(x)
    assert logits.shape == (Action.n(),)
    assert value.shape == ()  # scalar


def test_forward_rejects_bad_shape() -> None:
    net = ActorCriticNet(hidden_size=32)
    with pytest.raises(ValueError):
        net(torch.randn(4, 10))


def test_actor_softmax_sums_to_one() -> None:
    net = ActorCriticNet(hidden_size=32)
    logits, _ = net(torch.randn(8, net.state_dim))
    probs = nn_f.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)


def test_outputs_are_finite() -> None:
    torch.manual_seed(0)
    net = ActorCriticNet(hidden_size=16)
    logits, value = net(torch.randn(16, net.state_dim))
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_actor_and_critic_params_partition_trunk() -> None:
    """Trunk belongs to actor optimizer; critic optimizer sees only its head."""
    net = ActorCriticNet(hidden_size=16)
    actor_ids = {id(p) for p in net.actor_params()}
    critic_ids = {id(p) for p in net.critic_params()}
    trunk_ids = {id(p) for p in net.trunk.parameters()}
    actor_head_ids = {id(p) for p in net.actor_head.parameters()}
    critic_head_ids = {id(p) for p in net.critic_head.parameters()}
    assert trunk_ids.issubset(actor_ids)
    assert actor_head_ids.issubset(actor_ids)
    assert critic_head_ids == critic_ids
    # Trunk must NOT be under the critic optimizer (would double-step it).
    assert actor_ids.isdisjoint(critic_ids)
