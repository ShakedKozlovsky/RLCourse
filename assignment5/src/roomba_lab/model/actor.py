"""Deterministic actor μ(s | θ_μ) — slide 5 of L09.

Returns the action vector directly via `tanh` → output always in `[−1, 1]^d`. This
is the network the spec § Item 1 wants pointed at: 'the deterministic action
output through tanh function'."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from roomba_lab.model.init import init_actor_head, init_hidden


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layer = nn.Linear(prev, h)
            init_hidden(layer)
            layers.extend([layer, nn.ReLU()])
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, action_dim)
        init_actor_head(self.head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(obs)))
