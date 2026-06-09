"""GaussianActor — forward distribution + log_std clamping."""

from __future__ import annotations

import pytest
import torch

from proximal_lab.model.actor import GaussianActor


def test_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        GaussianActor(obs_dim=0, action_dim=6)
    with pytest.raises(ValueError):
        GaussianActor(obs_dim=17, action_dim=0)


def test_forward_returns_distribution() -> None:
    actor = GaussianActor(obs_dim=17, action_dim=6)
    obs = torch.randn(8, 17)
    dist = actor(obs)
    assert dist.mean.shape == (8, 6)
    assert dist.stddev.shape == (8, 6)


def test_sample_then_log_prob_shapes() -> None:
    torch.manual_seed(0)
    actor = GaussianActor(obs_dim=17, action_dim=6)
    obs = torch.randn(4, 17)
    dist = actor(obs)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    assert log_prob.shape == (4,)
    assert torch.isfinite(log_prob).all()


def test_log_std_clamped_above_max() -> None:
    actor = GaussianActor(obs_dim=4, action_dim=2,
                           log_std_init=10.0, log_std_max=2.0)
    obs = torch.zeros(1, 4)
    dist = actor(obs)
    # log_std clamped to 2 → stddev = e^2 ≈ 7.389
    assert torch.allclose(dist.stddev, torch.full_like(dist.stddev, torch.exp(torch.tensor(2.0))))


def test_log_std_clamped_below_min() -> None:
    actor = GaussianActor(obs_dim=4, action_dim=2,
                           log_std_init=-10.0, log_std_min=-5.0)
    obs = torch.zeros(1, 4)
    dist = actor(obs)
    assert torch.allclose(dist.stddev, torch.full_like(dist.stddev, torch.exp(torch.tensor(-5.0))))


def test_log_prob_differentiable_wrt_actor_params() -> None:
    actor = GaussianActor(obs_dim=4, action_dim=2)
    obs = torch.randn(3, 4, requires_grad=False)
    dist = actor(obs)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()
    log_prob.backward()
    assert any(p.grad is not None for p in actor.parameters())
