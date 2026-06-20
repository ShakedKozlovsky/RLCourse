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
    """Deterministic actor for DDPG.

    The `head_gain` parameter controls the orthogonal-init scale of the final
    layer (Layer 23 made it config-driven per TA Mod7). Larger gains give the
    initial policy more action magnitude, which matters when the env penalises
    inaction. For purely-discrete-spec DDPG (Lillicrap 2016) a near-zero gain
    (~0.003) is fine; for our cleaning-robot env we default to 0.1 so the
    initial actor produces meaningful forward velocity from step 0."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        head_gain: float = 0.1,
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
        init_actor_head(self.head, gain=head_gain)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: return tanh-bounded action in [-1, +1]^action_dim."""
        return torch.tanh(self.head(self.body(obs)))
