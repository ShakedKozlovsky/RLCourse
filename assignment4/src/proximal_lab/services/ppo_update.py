"""PPO update step + diagnostics aggregation — factored out of ``ppo_service.py``."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.services.ppo_clip import approx_kl, ppo_clip_loss
from proximal_lab.services.rollout_buffer import RolloutBuffer


def ppo_update(
    net: ActorCriticNet,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    *,
    clip_eps: float,
    n_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
) -> dict[str, float]:
    """Run K-epoch minibatch PPO update on a filled buffer; return diagnostics."""
    net.train()
    rng = np.random.default_rng(0)
    kls: list[float] = []
    clip_fracs: list[float] = []
    value_losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    for _ in range(n_epochs):
        early_stop = False
        for mb in buf.minibatches(minibatch_size, rng=rng):
            kl_step, clip_step, vl, pl, ent = _step(
                net, opt, mb, clip_eps=clip_eps, value_coef=value_coef,
                entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
            )
            kls.append(kl_step)
            clip_fracs.append(clip_step)
            value_losses.append(vl)
            policy_losses.append(pl)
            entropies.append(ent)
            if target_kl is not None and kl_step > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break
    explained_var = _explained_variance(buf)
    return {
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "explained_variance": explained_var,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def _step(
    net: ActorCriticNet,
    opt: optim.Optimizer,
    mb: dict[str, torch.Tensor],
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tuple[float, float, float, float, float]:
    """One minibatch SGD step; returns ``(kl, clip_frac, v_loss, p_loss, entropy)``."""
    advantages = mb["advantages"]
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    new_lp, new_entropy, new_values = net.evaluate(mb["observations"], mb["actions"])
    ratio = torch.exp(new_lp - mb["log_probs_old"])
    policy_loss, clip_frac = ppo_clip_loss(ratio, advantages, clip_eps)
    value_loss = 0.5 * (new_values - mb["returns"]).pow(2).mean()
    entropy = new_entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    opt.step()
    with torch.no_grad():
        kl = float(approx_kl(new_lp, mb["log_probs_old"]).item())
    return (kl, float(clip_frac.item()), float(value_loss.item()),
            float(policy_loss.item()), float(entropy.item()))


def _explained_variance(buf: RolloutBuffer) -> float:
    returns = torch.from_numpy(buf.returns.reshape(-1)).float()
    values = torch.from_numpy(buf.values.reshape(-1)).float()
    return float(1.0 - (returns - values).var() / (returns.var() + 1e-8))
