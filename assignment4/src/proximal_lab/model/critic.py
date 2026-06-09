"""Value-function critic — MLP returning scalar V(s).

See ``docs/PRD_actor_critic.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from proximal_lab.model.init import CRITIC_HEAD_GAIN, init_mlp


class Critic(nn.Module):
    """obs → tanh-MLP → scalar V(s)."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        if obs_dim < 1:
            raise ValueError("obs_dim must be >= 1")
        sizes = [obs_dim, *hidden_sizes, 1]
        layers: list[nn.Linear] = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
        init_mlp(layers, head_gain=CRITIC_HEAD_GAIN)
        self._layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = obs
        for layer in self._layers[:-1]:
            h = torch.tanh(layer(h))
        return self._layers[-1](h).squeeze(-1)
