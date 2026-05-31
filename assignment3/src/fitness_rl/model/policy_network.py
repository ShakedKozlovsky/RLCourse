"""Policy network for REINFORCE — 2-layer MLP returning action logits.

Same actor architecture as the A2C actor head so REINFORCE-vs-A2C
comparison isolates the *learning rule* from network capacity.
See ``docs/PRD_reinforce.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from fitness_rl.data.feature_engineer import STATE_DIM
from fitness_rl.shared.types import Action


class PolicyNet(nn.Module):
    """MLP ``state -> logits``: D_s -> H -> H -> n_actions."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        self.state_dim = STATE_DIM
        self.n_actions = Action.n()
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``state`` is (B, D_s) or (D_s,); returns logits of matching batch shape."""
        if state.dim() == 1:
            return self.net(state.unsqueeze(0)).squeeze(0)
        if state.dim() == 2 and state.size(-1) == self.state_dim:
            return self.net(state)
        raise ValueError(f"expected (B, {self.state_dim}) or ({self.state_dim},); got {tuple(state.shape)}")
