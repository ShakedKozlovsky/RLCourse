"""Gaussian actor for continuous action spaces — μ(s) MLP + state-independent log_std.

See ``docs/PRD_actor_critic.md`` for the rationale.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from proximal_lab.model.init import ACTOR_HEAD_GAIN, init_mlp


class GaussianActor(nn.Module):
    """μ(s) = MLP(obs); log_std is a separate learned parameter vector."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if obs_dim < 1 or action_dim < 1:
            raise ValueError("obs_dim and action_dim must be >= 1")
        sizes = [obs_dim, *hidden_sizes, action_dim]
        layers: list[nn.Linear] = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
        init_mlp(layers, head_gain=ACTOR_HEAD_GAIN)
        self._layers = nn.ModuleList(layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: torch.Tensor) -> Normal:
        h = obs
        for layer in self._layers[:-1]:
            h = torch.tanh(layer(h))
        mu = self._layers[-1](h)
        clamped = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = clamped.exp().expand_as(mu)
        return Normal(mu, std)
