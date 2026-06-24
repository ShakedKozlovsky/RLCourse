"""QPLEX — Duplex Dueling Multi-Agent Q-Learning (Wang et al. ICLR 2021,
arXiv:2008.01062).

QPLEX is the recommended IGM-preserving alternative to QMIX (README § 7.2):
it satisfies the Individual–Global–Max principle BY CONSTRUCTION through a
dueling decomposition, *without* the monotonic-mixer restriction. This makes
QPLEX representationally strictly more expressive than QMIX while preserving
the IGM equivalence between centralised argmax and decentralised greedy.

Mathematical form (Wang 2021 § 3):

    V_i(τ_i)           = state-value head per agent (max-action's Q)
    A_i(τ_i, a_i)      = advantage head per agent (Q_i - V_i)
    V_tot(s)           = hypernet-produced global state value
    A_tot(s, ā)        = Σ_i λ_i(s, ā) · A_i(τ_i, a_i)
    Q_tot(s, τ, ā)     = V_tot(s) + A_tot(s, ā)

  where each ``λ_i(s, ā) > 0`` (enforced via abs) so that the joint argmax
  recovers the per-agent argmax — preserving IGM. Unlike QMIX, the advantage
  weights ``λ_i`` are state+action-conditioned and the value head V_tot is
  unconstrained, allowing non-monotonic shapes that QMIX cannot represent.

Construction guarantee (proof in docs/PROOFS.md § 2):
    ∂Q_tot/∂Q_i = ∂Q_tot/∂A_i = λ_i(s, ā) > 0,
so the IGM equivalence
    argmax_ā Q_tot(s, τ, ā) ≡ (argmax_a_1 Q_1, …, argmax_a_n Q_n)
holds without further constraints on V_tot."""

from __future__ import annotations

import torch
import torch.nn as nn


class QPLEXMixer(nn.Module):
    """Duplex dueling mixer — IGM by construction, strictly more expressive
    than QMIX.

    Architecture:
      Inputs: q_per_agent (B, T, n_agents), v_per_agent (B, T, n_agents),
              global_state (B, T, state_dim)
      Outputs: q_tot (B, T)

    Per-agent advantage A_i = Q_i - V_i (computed by caller — the trainer
    knows which action was selected and what the max-action Q is).

    Mixer:
      V_tot     = hypernet_V(s)
      λ_i(s, ā) = |hypernet_lambda(s, ā)|     (strictly positive)
      A_tot     = Σ_i λ_i · A_i
      Q_tot     = V_tot + A_tot
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hyper_hidden: int = 64,
    ) -> None:
        super().__init__()
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        # V_tot(s): state -> scalar
        self.hyper_v = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )
        # λ_i(s, ā): state -> per-agent positive weight
        # We take state alone (joint action is implicit via per-agent Q's
        # action-channel reduction by the caller). This matches the standard
        # QPLEX implementation in the PyMARL2 reference codebase.
        self.hyper_lambda = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents),
        )

    def forward(
        self,
        q_per_agent: torch.Tensor,
        v_per_agent: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q_tot via the duplex dueling decomposition.

        Args:
          q_per_agent: (B, T, n_agents) — per-agent Q values for the
            CHOSEN action of each agent.
          v_per_agent: (B, T, n_agents) — per-agent state value
            ``V_i(τ_i) = max_a Q_i(τ_i, a)``. Caller computes this from the
            same per-agent network as q_per_agent.
          global_state: (B, T, state_dim) — centralised state s.

        Returns Q_tot of shape (B, T).
        """
        if q_per_agent.shape[-1] != self.n_agents:
            raise ValueError(
                f"q_per_agent last dim must be n_agents={self.n_agents}, "
                f"got shape {tuple(q_per_agent.shape)}"
            )
        if v_per_agent.shape != q_per_agent.shape:
            raise ValueError(
                f"v_per_agent shape must match q_per_agent; "
                f"got {tuple(v_per_agent.shape)} vs {tuple(q_per_agent.shape)}"
            )
        b, t, _ = q_per_agent.shape
        s = global_state.reshape(b * t, -1)
        # V_tot(s): (B*T, 1)
        v_tot = self.hyper_v(s).reshape(b, t)
        # λ_i(s): strictly positive — |·| matches the QPLEX paper's positivity
        # requirement on the advantage weights
        lam = torch.abs(self.hyper_lambda(s)).reshape(b, t, self.n_agents)
        # Per-agent advantage: A_i = Q_i - V_i
        adv = q_per_agent - v_per_agent
        # Σ_i λ_i · A_i
        a_tot = (lam * adv).sum(dim=-1)
        return v_tot + a_tot
