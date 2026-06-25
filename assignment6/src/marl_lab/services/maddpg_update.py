"""MADDPG-discrete update step — per-agent centralised critics with
per-agent reward (true POSG learner, no averaging hack).

Compared to qmix_update.py:
  - NO mixer (no `Q_tot`). Instead, ONE critic per agent.
  - Each critic Q_i^C(s, ā) is trained with that agent's OWN reward.
  - The local per-agent Q-net is regressed toward the centralised critic's
    value (knowledge-distillation-style), so execution stays decentralised
    while still benefiting from the centralised training signal."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.model.maddpg_critic import MADDPGCritic, one_hot_joint_action
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.soft_update import polyak_update

AGENTS = ("cop", "thief")
N_PER_AGENT = {"cop": 6, "thief": 5}     # spec § 3.1 + § 3.3


@dataclass(frozen=True)
class MaddpgUpdateDiagnostic:
    """Per-update logging tuple — exposes per-agent critic losses since
    MADDPG has no joint Q_tot to log."""
    critic_loss_cop: float
    critic_loss_thief: float
    mean_q_cop: float
    mean_q_thief: float
    target_drift: float


def _batch_to_tensors(batch: BatchTensors, device: torch.device) -> dict:
    return {
        "state": torch.as_tensor(batch.state_seq, device=device),
        "next_state": torch.as_tensor(batch.next_state_seq, device=device),
        "obs": {a: torch.as_tensor(batch.obs_seq[a], device=device) for a in AGENTS},
        "next_obs": {a: torch.as_tensor(batch.next_obs_seq[a], device=device) for a in AGENTS},
        "action": {a: torch.as_tensor(batch.action_seq[a], device=device) for a in AGENTS},
        "reward": {a: torch.as_tensor(batch.reward_seq[a], device=device) for a in AGENTS},
        "done": torch.as_tensor(batch.done_seq, device=device),
        "mask": torch.as_tensor(batch.mask, device=device),
    }


def apply_maddpg_update(
    q_nets: dict[str, QPerAgent],
    target_q_nets: dict[str, QPerAgent],
    critics: dict[str, MADDPGCritic],
    target_critics: dict[str, MADDPGCritic],
    batch: BatchTensors,
    gamma: float,
    tau: float,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> MaddpgUpdateDiagnostic:
    """One MADDPG-discrete update step.

    Per-agent steps:
      1. critic Q_i^C(s, ā) forward with the buffer's chosen joint action.
      2. target ā' = (argmax_a Q_cop^local(o'_cop), argmax_a Q_thief^local(o'_thief)).
      3. target_i = r_i + γ Q_i^C_target(s', ā').
      4. critic loss_i = MSE(Q_i^C(s, ā), target_i) * mask.
      5. local Q-net regressed: L_local_i = MSE(Q_i^local(o_i, a_i),
         Q_i^C(s, ā).detach()) * mask. This is the CTDE-distillation step.

    All per-agent losses are summed and backproped through a single optimiser
    that holds the parameters of BOTH the local Q-nets AND the critics.
    Target nets are then Polyak-updated at rate τ."""
    device = device or torch.device("cpu")
    b = _batch_to_tensors(batch, device)
    target_before = torch.cat(
        [p.data.flatten() for a in AGENTS for p in target_critics[a].parameters()]
        + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
    ).clone()

    # ----- ONE-HOT joint actions for the critic -----
    joint_oh = one_hot_joint_action(
        actions_per_agent={a: b["action"][a] for a in AGENTS},
        n_actions_per_agent=N_PER_AGENT,
    )                                              # (B, T, n_total)

    # ----- Forward live Q-nets, gather chosen-action Q -----
    q_local_chosen: dict[str, torch.Tensor] = {}
    for a in AGENTS:
        q_seq, _ = q_nets[a](b["obs"][a], hidden=None)
        a_idx = b["action"][a].unsqueeze(-1)
        q_local_chosen[a] = q_seq.gather(dim=-1, index=a_idx).squeeze(-1)

    # ----- Target joint action ā' from local Q-net argmax (no grad) -----
    with torch.no_grad():
        next_target_actions: dict[str, torch.Tensor] = {}
        for a in AGENTS:
            n_legal = N_PER_AGENT[a]
            q_next, _ = target_q_nets[a](b["next_obs"][a], hidden=None)
            # Mask illegal actions for the thief (no PLACE_BARRIER)
            q_next_masked = q_next.clone()
            q_next_masked[..., n_legal:] = float("-inf")
            next_target_actions[a] = q_next_masked.argmax(dim=-1)
        next_joint_oh = one_hot_joint_action(
            actions_per_agent=next_target_actions,
            n_actions_per_agent=N_PER_AGENT,
        )

    # ----- Per-agent losses -----
    mask = b["mask"]
    losses: dict[str, torch.Tensor] = {}
    distill_losses: dict[str, torch.Tensor] = {}
    for a in AGENTS:
        # Live critic
        q_c = critics[a](b["state"], joint_oh)                            # (B, T)
        with torch.no_grad():
            q_c_next = target_critics[a](b["next_state"], next_joint_oh)
            target_i = b["reward"][a] + gamma * (1.0 - b["done"]) * q_c_next
        critic_loss = (((q_c - target_i) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        losses[a] = critic_loss
        # Distil local Q-net toward centralised critic value (detached so
        # the critic isn't double-updated via this path)
        distill = (((q_local_chosen[a] - q_c.detach()) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        distill_losses[a] = distill

    total_loss = sum(losses.values()) + sum(distill_losses.values())

    # ----- Backprop + step -----
    critic_opt.zero_grad()
    total_loss.backward()
    all_params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        all_params.extend(q_nets[a].parameters())
        all_params.extend(critics[a].parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    critic_opt.step()

    # ----- Polyak target updates -----
    for a in AGENTS:
        polyak_update(target_critics[a].parameters(), critics[a].parameters(), tau)
        polyak_update(target_q_nets[a].parameters(), q_nets[a].parameters(), tau)

    # ----- Diagnostics -----
    with torch.no_grad():
        means: dict[str, float] = {}
        for a in AGENTS:
            means[a] = float((q_local_chosen[a] * mask).sum().item()
                              / mask.sum().clamp(min=1.0))
        target_after = torch.cat(
            [p.data.flatten() for a in AGENTS for p in target_critics[a].parameters()]
            + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
        )
        drift = float((target_after - target_before).abs().mean().item())
    return MaddpgUpdateDiagnostic(
        critic_loss_cop=float(losses["cop"].item()),
        critic_loss_thief=float(losses["thief"].item()),
        mean_q_cop=means["cop"],
        mean_q_thief=means["thief"],
        target_drift=drift,
    )
