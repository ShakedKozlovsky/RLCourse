"""TD3 update step — twin-critic + delayed actor + target-noise smoothing.

Three TD3 features (Fujimoto 2018):
  1. Bootstrap target = r + γ (1-d) min(Q'_a(s', μ'+ε), Q'_b(s', μ'+ε))
  2. Actor update only every `policy_delay` critic steps
  3. ε ~ N(0, σ_target) clipped to [-c, c] added to target action"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from roomba_lab.model.soft_update import polyak_update
from roomba_lab.model.td3_network import TD3Network


@dataclass(frozen=True)
class TD3Diagnostic:
    critic_loss: float
    actor_loss: float
    mean_q: float
    actor_updated: bool


def critic_loss_td3(net: TD3Network, batch: dict, gamma: float,
                    target_policy_noise: float, target_noise_clip: float) -> torch.Tensor:
    """Critic loss td3."""
    b = {k: torch.as_tensor(v) for k, v in batch.items()}
    with torch.no_grad():
        next_action = net.target_actor(b["next_state"])
        epsilon = (torch.randn_like(next_action) * target_policy_noise).clamp(
            -target_noise_clip, target_noise_clip
        )
        smoothed = (next_action + epsilon).clamp(-1.0, 1.0)
        q_a = net.target_critic_a(b["next_state"], smoothed)
        q_b = net.target_critic_b(b["next_state"], smoothed)
        q_next = torch.min(q_a, q_b)
        y = b["reward"] + gamma * (1.0 - b["done"]) * q_next
    q_pred_a = net.critic_a(b["state"], b["action"])
    q_pred_b = net.critic_b(b["state"], b["action"])
    return functional.mse_loss(q_pred_a, y) + functional.mse_loss(q_pred_b, y)


def actor_loss_td3(net: TD3Network, batch: dict) -> torch.Tensor:
    """Actor loss td3."""
    b = {k: torch.as_tensor(v) for k, v in batch.items()}
    return -net.critic_a(b["state"], net.actor(b["state"])).mean()


def apply_td3_update(
    net: TD3Network,
    batch: dict,
    step: int,
    gamma: float,
    tau: float,
    policy_delay: int,
    target_policy_noise: float,
    target_noise_clip: float,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
) -> TD3Diagnostic:
    """Apply td3 update."""
    c_loss = critic_loss_td3(net, batch, gamma, target_policy_noise, target_noise_clip)
    critic_opt.zero_grad()
    c_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(net.critic_a.parameters()) + list(net.critic_b.parameters()), max_grad_norm
    )
    critic_opt.step()
    actor_updated = (step % max(1, policy_delay)) == 0
    a_loss_val = 0.0
    if actor_updated:
        a_loss = actor_loss_td3(net, batch)
        actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.actor.parameters(), max_grad_norm)
        actor_opt.step()
        a_loss_val = float(a_loss.item())
        polyak_update(net.target_actor.parameters(), net.actor.parameters(), tau)
        polyak_update(net.target_critic_a.parameters(), net.critic_a.parameters(), tau)
        polyak_update(net.target_critic_b.parameters(), net.critic_b.parameters(), tau)
    with torch.no_grad():
        s = torch.as_tensor(batch["state"])
        mean_q = float(net.critic_a(s, net.actor(s)).mean().item())
    return TD3Diagnostic(critic_loss=float(c_loss.item()), actor_loss=a_loss_val,
                          mean_q=mean_q, actor_updated=actor_updated)
