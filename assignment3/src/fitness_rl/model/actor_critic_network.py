"""Actor-critic network for A2C — shared trunk + actor head + critic head.

Shared MLP trunk lets the actor and critic re-use the same feature
representation (lecture slide 20). The two heads diverge: actor → logits
over actions, critic → scalar state value V_ψ(s).
See ``docs/PRD_a2c.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from fitness_rl.data.feature_engineer import STATE_DIM
from fitness_rl.shared.types import Action


class ActorCriticNet(nn.Module):
    """Shared trunk D_s → H → H, then actor (H → n_actions) + critic (H → 1)."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        self.state_dim = STATE_DIM
        self.n_actions = Action.n()
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size, self.n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)``; shapes match input batch dim."""
        was_single = state.dim() == 1
        if was_single:
            state = state.unsqueeze(0)
        if state.dim() != 2 or state.size(-1) != self.state_dim:
            raise ValueError(f"expected (B, {self.state_dim}) or ({self.state_dim},); got {tuple(state.shape)}")
        h = self.trunk(state)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        if was_single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    def actor_params(self) -> list[torch.nn.Parameter]:
        """Trunk + actor-head params — fed to the actor optimizer.

        The trunk lives under the actor so it is stepped exactly once per
        update. The critic head still receives gradients from `critic_loss`
        through the trunk, but the trunk weights are only nudged by `actor_lr`.
        """
        return list(self.trunk.parameters()) + list(self.actor_head.parameters())

    def critic_params(self) -> list[torch.nn.Parameter]:
        """Critic-head params only — fed to the critic optimizer."""
        return list(self.critic_head.parameters())
