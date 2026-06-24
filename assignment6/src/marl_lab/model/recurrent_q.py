"""Per-agent recurrent Q-network — Qᵢ(oᵢ_t, hᵢ_t) → action values.

L10 § 5: GRU over the local observation sequence. Partial observability →
the agent must integrate past observations to infer the global state."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from marl_lab.model.init import init_hidden, init_q_head


class QPerAgent(nn.Module):
    """Per-agent recurrent Q-network with a GRU layer.

    Architecture:
      obs (B, T, obs_dim) → [Linear, ReLU]+ → GRU → Linear → q (B, T, n_actions)

    The GRU hidden state ``h`` is propagated across timesteps within an episode
    (the CTDE trainer carries it; tests can pass ``h=None`` for the first call)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        gru_hidden_size: int = 64,
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
        self.gru = nn.GRU(input_size=prev, hidden_size=gru_hidden_size,
                          num_layers=1, batch_first=True)
        self.head = nn.Linear(gru_hidden_size, n_actions)
        init_q_head(self.head)
        self.gru_hidden_size = gru_hidden_size

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a sequence.

        Args:
          obs: shape (B, T, obs_dim) — batched sequence of observations.
          hidden: shape (1, B, gru_hidden_size) — initial GRU state. If None,
            zeros are used (start of an episode).

        Returns:
          q_values: shape (B, T, n_actions)
          new_hidden: shape (1, B, gru_hidden_size) — to chain into next call."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, obs_dim) → (B, 1, obs_dim)
        b, t, _ = obs.shape
        # Body operates per-step
        flat = obs.reshape(b * t, -1)
        feats = self.body(flat).reshape(b, t, -1)
        if hidden is None:
            hidden = torch.zeros(1, b, self.gru_hidden_size,
                                  device=obs.device, dtype=obs.dtype)
        seq_out, new_hidden = self.gru(feats, hidden)
        q = self.head(seq_out)
        return q, new_hidden

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Zero initial hidden state of shape (1, batch_size, gru_hidden_size)."""
        return torch.zeros(1, batch_size, self.gru_hidden_size,
                            device=device or torch.device("cpu"))
