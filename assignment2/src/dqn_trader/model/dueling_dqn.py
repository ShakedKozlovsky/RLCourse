"""Dueling DQN (and vanilla DQN fallback) over a Conv1D feature extractor.

Architecture matches PLAN §6 / PRD_dueling.md. Shared Conv trunk feeds a
state-value head V(s) and an advantage head A(s, a); the final Q is the
mean-centred combination ``V(s) + (A − mean A)``. When ``dueling=False``
the heads collapse to a single linear projection — the *vanilla DQN* used
in the comparative experiment.
"""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    """(B, T, F) → (B, n_actions). Set ``dueling=False`` for vanilla DQN."""

    def __init__(
        self,
        window_size: int,
        n_features: int,
        n_actions: int,
        hidden: int = 128,
        *,
        dueling: bool = True,
    ):
        super().__init__()
        if window_size < 2 or n_features < 1 or n_actions < 2:
            raise ValueError("invalid network shape")
        self.window_size = window_size
        self.n_features = n_features
        self.n_actions = n_actions
        self.dueling = dueling
        self.trunk = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat = 64 * window_size
        self.shared_fc = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU())
        if dueling:
            self.value_head = nn.Linear(hidden, 1)
            self.advantage_head = nn.Linear(hidden, n_actions)
        else:
            self.q_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F), got shape {tuple(x.shape)}")
        # Conv1d expects (B, C, T); we have (B, T, F) so permute features → channels.
        feats = self.trunk(x.permute(0, 2, 1))
        h = self.shared_fc(feats)
        if not self.dueling:
            return self.q_head(h)
        v = self.value_head(h)  # (B, 1)
        a = self.advantage_head(h)  # (B, n_actions)
        # Mean-centred advantage stream removes the additive ambiguity in V/A decomposition.
        return v + (a - a.mean(dim=1, keepdim=True))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average ``target ← tau · source + (1 − tau) · target``."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy source weights to target. Used at fixed-interval target sync."""
    target.load_state_dict(source.state_dict())
