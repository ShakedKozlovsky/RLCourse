"""Generalized Advantage Estimation — Schulman et al. 2016.

Reverse recursion form (Eq. 2c in ``docs/PRD_gae.md``):

    δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    Â_t = δ_t + γ · λ · (1 − done_t) · Â_{t+1}

This is a **pure function** — no state, no side effects, no logger.
Standalone module so it's trivially testable.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: float,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Compute GAE advantages over one rollout.

    Parameters
    ----------
    rewards : shape (T,)
        Per-step rewards.
    values : shape (T,)
        Per-step critic predictions V(s_t).
    last_value : scalar
        Bootstrap V(s_T) — the critic's prediction for the obs after the last step.
    dones : shape (T,) bool
        ``dones[t]`` is True if the episode ended *at step t* (the transition
        from s_t to s_{t+1} was terminal).
    gamma : float in (0, 1]
        Discount factor.
    lam : float in [0, 1]
        GAE bias-variance dial. λ=0 → TD error, λ=1 → Monte-Carlo − V.

    Returns
    -------
    advantages : shape (T,) float32
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, values {values.shape}, "
            f"dones {dones.shape}"
        )
    t = rewards.shape[0]
    advantages = np.zeros(t, dtype=np.float32)
    gae = 0.0
    next_value = float(last_value)
    for step in reversed(range(t)):
        next_non_terminal = 1.0 - float(dones[step])
        delta = float(rewards[step]) + gamma * next_value * next_non_terminal - float(values[step])
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[step] = gae
        next_value = float(values[step])
    return advantages
