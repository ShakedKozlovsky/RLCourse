"""VDN — Value-Decomposition Network (Sunehag et al. AAMAS 2018).

The simplest CTDE mixer:  Q_tot = Σ Qᵢ. Trivially satisfies the IGM principle
(each Qᵢ is monotone in itself; the sum is monotone in each Qᵢ)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    """Additive value decomposition.

    Stateless — no parameters. Kept as an nn.Module so it composes with
    optimisers and Polyak updates the same way QMIX does."""

    def __init__(self, n_agents: int) -> None:
        super().__init__()
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        self.n_agents = int(n_agents)

    def forward(self, q_per_agent: torch.Tensor, global_state: torch.Tensor | None = None) -> torch.Tensor:
        """Sum per-agent Q-values along the agent dimension.

        Args:
          q_per_agent: shape (..., n_agents) — per-agent Q values for the
            chosen action of each agent.
          global_state: ignored; accepted for interface compatibility with
            QMIXMixer (so the training loop is mixer-agnostic).

        Returns:
          Q_tot of shape (...).
        """
        if q_per_agent.shape[-1] != self.n_agents:
            raise ValueError(
                f"expected last dim == n_agents ({self.n_agents}), "
                f"got {tuple(q_per_agent.shape)}"
            )
        return q_per_agent.sum(dim=-1)
