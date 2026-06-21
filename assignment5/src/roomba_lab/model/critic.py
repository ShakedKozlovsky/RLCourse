"""Critic Q(s, a | θ_Q) — slide 5 of L09.

The state and action are **concatenated at the input layer** per slide 5 (and
per Lillicrap 2016 — though that paper put the action in at layer 2; the
concat-at-input variant is just as common in production code and simpler)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from roomba_lab.model.init import init_critic_head, init_hidden


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim + action_dim
        for h in hidden_sizes:
            layer = nn.Linear(prev, h)
            init_hidden(layer)
            layers.extend([layer, nn.ReLU()])
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        init_critic_head(self.head)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Concat state + action at input layer, return scalar Q(s, a)."""
        x = torch.cat([obs, action], dim=-1)
        return self.head(self.body(x)).squeeze(-1)
