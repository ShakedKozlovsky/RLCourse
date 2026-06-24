"""Polyak (soft) target update — slide 6 of L09 / standard DDPG.

θ_target ← τ · θ_source + (1 − τ) · θ_target. Carried over from A5 with the
same 4-test math battery. Used by all three MARL updaters (QMIX, VDN, IQL)."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def polyak_update(
    target_params: Iterable[torch.nn.Parameter],
    source_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """θ_target ← τ·θ_source + (1−τ)·θ_target — slide 6 Polyak soft-target average.

    In-place under ``torch.no_grad()`` so target params stay disconnected from
    autograd. τ=0 freezes the target; τ=1 hard-copies; τ=0.005 (default)
    gives ~200-step exponential smoothing."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    with torch.no_grad():
        for t, s in zip(target_params, source_params, strict=True):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)


def hard_copy(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Equivalent of polyak_update(tau=1.0); used at init + hard-copy ablation."""
    target.load_state_dict(source.state_dict())
