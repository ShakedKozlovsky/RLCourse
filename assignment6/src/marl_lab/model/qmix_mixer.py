"""QMIX — Monotonic Value Function Factorisation (Rashid et al. ICML 2018).

Hypernetwork conditioned on the global state produces the mixer weights.
Monotonicity ∂Q_tot/∂Qᵢ ≥ 0 enforced via ``|·|`` (absolute value) on the
mixer weights — preserves IGM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  -- standard PyTorch alias


class QMIXMixer(nn.Module):
    """Two-layer monotonic mixer.

    Architecture::

        q_per_agent (B, T, n_agents)      global_state (B, T, state_dim)
                │                                       │
                │                            ┌──────────┴──────────┐
                │                            │   hypernet  W1, b1   │
                │                            │   hypernet  W2, b2   │
                │                            └──────────┬──────────┘
                ▼                                       │
         hidden = |W1| @ q + b1   ← monotone in each qᵢ
         ELU activation                                 │
         q_tot  = |W2| @ hidden + b2                    │
                                                        ▼

    The hypernet outputs ``W1, W2, b1, b2`` whose **absolute value** is used
    in the mixer — that's what guarantees ``∂Q_tot/∂Qᵢ ≥ 0``. The biases
    ``b1, b2`` can be negative (only the *weights* are constrained)."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hyper_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        # Hypernet for mixer layer 1: state → (n_agents × embed_dim) weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * embed_dim),
        )
        # Hypernet for layer 2: state → (embed_dim × 1) weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        # Biases are unconstrained (only weights need monotonicity)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, q_per_agent: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """Compute Q_tot conditioned on the global state.

        Args:
          q_per_agent: (B, T, n_agents) — per-agent Q values for the chosen
            actions.
          global_state: (B, T, state_dim) — centralised state.

        Returns:
          Q_tot of shape (B, T).
        """
        if q_per_agent.shape[-1] != self.n_agents:
            raise ValueError(
                f"expected last dim == n_agents ({self.n_agents}), "
                f"got {tuple(q_per_agent.shape)}"
            )
        b, t, _ = q_per_agent.shape
        s_flat = global_state.reshape(b * t, -1)
        q_flat = q_per_agent.reshape(b * t, 1, self.n_agents)
        # Layer 1: |W1|, b1
        w1 = torch.abs(self.hyper_w1(s_flat)).reshape(b * t, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(s_flat).reshape(b * t, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_flat, w1) + b1)        # (BT, 1, embed_dim)
        # Layer 2: |W2|, b2
        w2 = torch.abs(self.hyper_w2(s_flat)).reshape(b * t, self.embed_dim, 1)
        b2 = self.hyper_b2(s_flat).reshape(b * t, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                 # (BT, 1, 1)
        return q_tot.reshape(b, t)
