"""Polyak (soft) target update — slide 6 of L09, EX05 § Item 2.

θ_target ← τ · θ_source + (1 − τ) · θ_target.

Performed in-place under ``torch.no_grad()`` so target params stay disconnected
from autograd. Three regimes: τ=0 freezes the target forever, τ=0.005 (default)
gives ~200-step exponential smoothing, τ=1 is a hard copy (= no target net)."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def polyak_update(
    target_params: Iterable[torch.nn.Parameter],
    source_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """Polyak update."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    with torch.no_grad():
        for t, s in zip(target_params, source_params, strict=True):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)


def hard_copy(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Equivalent of `polyak_update(tau=1.0)`; used at init + the hard-copy ablation."""
    target.load_state_dict(source.state_dict())
