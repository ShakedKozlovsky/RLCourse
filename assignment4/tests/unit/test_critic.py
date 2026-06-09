"""Critic — scalar V(s)."""

from __future__ import annotations

import pytest
import torch

from proximal_lab.model.critic import Critic


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        Critic(obs_dim=0)


def test_forward_shape_batched() -> None:
    critic = Critic(obs_dim=17)
    obs = torch.randn(8, 17)
    v = critic(obs)
    assert v.shape == (8,)


def test_forward_shape_single() -> None:
    critic = Critic(obs_dim=17)
    obs = torch.randn(17)
    v = critic(obs)
    assert v.shape == ()  # scalar


def test_value_differentiable() -> None:
    critic = Critic(obs_dim=4)
    obs = torch.randn(3, 4)
    v = critic(obs).sum()
    v.backward()
    assert any(p.grad is not None for p in critic.parameters())
