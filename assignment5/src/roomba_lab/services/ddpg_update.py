"""Single DDPG update step — slide 8 algorithmic flow.

Three pure-ish functions:

  * ``critic_loss(net, batch, gamma)`` — MSE between Q(s, a) and the bootstrapped
    TD target  y = r + γ (1−done) Q'(s', μ'(s'))           (slide 6)
  * ``actor_loss(net, batch)``         — − mean Q(s, μ(s))                 (slide 4)
  * ``apply_update(net, ...)``         — orchestrates the two optim steps + Polyak

Spec § Item 1 traceability: the actor-loss expression IS the deterministic
policy gradient via the autograd chain rule."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.model.soft_update import polyak_update


@dataclass(frozen=True)
class UpdateDiagnostic:
    critic_loss: float
    actor_loss: float
    mean_q: float
    target_drift: float


def _batch_to_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def critic_loss(net: ActorCriticNet, batch: dict, gamma: float,
                device: torch.device | None = None) -> torch.Tensor:
    """MSE between Q(s, a) and the bootstrapped target y = r + γ(1-d)Q'(s', μ'(s')).

    The target networks (target_actor, target_critic) are used under no_grad
    — slide 6 of L09. This is the spec § Item 2 target-network mechanism."""
    device = device or torch.device("cpu")
    b = _batch_to_tensors(batch, device)
    with torch.no_grad():
        a_next = net.target_actor(b["next_state"])
        q_next = net.target_critic(b["next_state"], a_next)
        y = b["reward"] + gamma * (1.0 - b["done"]) * q_next
    q_pred = net.critic(b["state"], b["action"])
    return functional.mse_loss(q_pred, y)


def actor_loss(net: ActorCriticNet, batch: dict,
               device: torch.device | None = None) -> torch.Tensor:
    """Deterministic Policy Gradient surrogate: −E[Q(s, μ(s))].

    Slide 4 of L09. The minus sign + autograd chain rule implements
    ∇θ μ · ∇a Q automatically — the actor maximises critic output."""
    device = device or torch.device("cpu")
    b = _batch_to_tensors(batch, device)
    return -net.critic(b["state"], net.actor(b["state"])).mean()


def apply_update(
    net: ActorCriticNet,
    batch: dict,
    gamma: float,
    tau: float,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> UpdateDiagnostic:
    """One full DDPG update: critic step → actor step → Polyak target updates.

    Returns an `UpdateDiagnostic` with the four headline numbers:
    critic_loss, actor_loss, mean_q (≈ value-of-current-policy), target_drift
    (mean |Δ target_critic|, sanity-check the Polyak step actually moved)."""
    device = device or torch.device("cpu")
    target_before = torch.cat([p.data.flatten() for p in net.target_critic.parameters()]).clone()
    c_loss = critic_loss(net, batch, gamma, device)
    critic_opt.zero_grad()
    c_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.critic.parameters(), max_grad_norm)
    critic_opt.step()
    a_loss = actor_loss(net, batch, device)
    actor_opt.zero_grad()
    a_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.actor.parameters(), max_grad_norm)
    actor_opt.step()
    polyak_update(net.target_actor.parameters(), net.actor.parameters(), tau)
    polyak_update(net.target_critic.parameters(), net.critic.parameters(), tau)
    target_after = torch.cat([p.data.flatten() for p in net.target_critic.parameters()])
    drift = float((target_after - target_before).abs().mean().item())
    with torch.no_grad():
        mean_q = float(net.critic(_batch_to_tensors(batch, device)["state"],
                                    net.actor(_batch_to_tensors(batch, device)["state"])).mean().item())
    return UpdateDiagnostic(
        critic_loss=float(c_loss.item()),
        actor_loss=float(a_loss.item()),
        mean_q=mean_q,
        target_drift=drift,
    )
