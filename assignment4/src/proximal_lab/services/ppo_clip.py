"""PPO clipped surrogate — the headline math (Schulman et al. 2017 Eq. 7).

L^CLIP(θ) = Ê_t[ min( r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]      (Eq. 1)

Standalone module so the math is trivially testable against the four
sign × clip-window cases in ``docs/PRD_ppo.md``.
"""

from __future__ import annotations

import torch


def ppo_clip_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped surrogate loss.

    Parameters
    ----------
    ratio : shape (N,)
        ``π_θ(a|s) / π_θ_old(a|s)`` per transition.
    advantages : shape (N,)
        GAE advantages, optionally normalised.
    clip_eps : float
        The ε in ``clip(r, 1−ε, 1+ε)``.

    Returns
    -------
    loss : scalar tensor — ``−mean(min(r·Â, clip(r)·Â))``
    clip_fraction : scalar tensor — fraction of transitions outside the clip window
    """
    if clip_eps <= 0:
        raise ValueError("clip_eps must be > 0")
    if ratio.shape != advantages.shape:
        raise ValueError(f"shape mismatch: ratio {ratio.shape}, advantages {advantages.shape}")
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate_unclipped = ratio * advantages
    surrogate_clipped = clipped * advantages
    loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()
    clip_fraction = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean()
    return loss, clip_fraction


def approx_kl(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
) -> torch.Tensor:
    """Approximate KL divergence per Schulman's blog: ``E[log(p_old) − log(p_new)]``.

    Numerically stable, unbiased for small KL. Used for the target-KL early-stop.
    """
    return (log_prob_old - log_prob_new).mean()
