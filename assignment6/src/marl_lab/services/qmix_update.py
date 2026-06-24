"""QMIX update step — the headline math for CTDE training (PRD_ctde § 5).

One call to ``apply_qmix_update`` performs:
  1. critic forward: per-agent Q(o_t, h) over the sequence, gather chosen action
  2. centralised mix: Q_tot_chosen = QMIXMixer(Q_i_chosen, s_t)
  3. target forward (no grad): per-agent target_Q.max over actions on next obs
  4. target mix:     Q_tot_max_next = QMIXMixer_target(Q_i_max_target, s_{t+1})
  5. TD target:      y = r + γ (1 − d) Q_tot_max_next
  6. loss:           MSE(Q_tot_chosen, y) * mask, then mean
  7. backprop:       critic_opt step
  8. Polyak target update: ALL targets (Q-net + Mixer) at rate τ"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.model.qmix_mixer import QMIXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.soft_update import polyak_update

AGENTS = ("cop", "thief")


@dataclass(frozen=True)
class QmixUpdateDiagnostic:
    """Per-update logging tuple."""
    critic_loss: float
    mean_q_cop: float
    mean_q_thief: float
    target_drift: float           # mean |Δ target_critic params|


def _batch_to_tensors(batch: BatchTensors, device: torch.device) -> dict:
    """Convert numpy BatchTensors to torch on the target device."""
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


def apply_qmix_update(
    q_nets: dict[str, QPerAgent],
    target_q_nets: dict[str, QPerAgent],
    mixer: QMIXMixer,
    target_mixer: QMIXMixer,
    batch: BatchTensors,
    gamma: float,
    tau: float,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> QmixUpdateDiagnostic:
    """One full QMIX update step (PRD_ctde § 5).

    The optimiser must include the parameters of ALL per-agent Q-nets AND the
    mixer. Returns diagnostics for logging."""
    device = device or torch.device("cpu")
    b = _batch_to_tensors(batch, device)
    # Snapshot target params for drift computation
    target_before = torch.cat(
        [p.data.flatten() for p in target_mixer.parameters()]
        + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
    ).clone()

    # === Per-agent forward (live Q nets) over the sequence ===
    q_chosen_per_agent: list[torch.Tensor] = []
    for a in AGENTS:
        q_seq, _ = q_nets[a](b["obs"][a], hidden=None)            # (B, T, n_actions)
        a_idx = b["action"][a].unsqueeze(-1)                       # (B, T, 1)
        q_chosen = q_seq.gather(dim=-1, index=a_idx).squeeze(-1)   # (B, T)
        q_chosen_per_agent.append(q_chosen)
    q_chosen_stack = torch.stack(q_chosen_per_agent, dim=-1)        # (B, T, n_agents)
    q_tot_chosen = mixer(q_chosen_stack, b["state"])                # (B, T)

    # === Target Q max (no grad) ===
    with torch.no_grad():
        q_max_per_agent: list[torch.Tensor] = []
        for a in AGENTS:
            q_next_seq, _ = target_q_nets[a](b["next_obs"][a], hidden=None)
            q_max = q_next_seq.max(dim=-1).values                  # (B, T)
            q_max_per_agent.append(q_max)
        q_max_stack = torch.stack(q_max_per_agent, dim=-1)         # (B, T, n_agents)
        q_tot_max_next = target_mixer(q_max_stack, b["next_state"])  # (B, T)
        # Joint reward (cooperative-treatment) — average of per-agent rewards
        joint_reward = (b["reward"]["cop"] + b["reward"]["thief"]) * 0.5
        y = joint_reward + gamma * (1.0 - b["done"]) * q_tot_max_next

    # === Loss with mask ===
    td_error = (q_tot_chosen - y) ** 2
    mask = b["mask"]
    loss = (td_error * mask).sum() / mask.sum().clamp(min=1.0)

    # === Critic step ===
    critic_opt.zero_grad()
    loss.backward()
    all_params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        all_params.extend(q_nets[a].parameters())
    all_params.extend(mixer.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    critic_opt.step()

    # === Polyak target updates ===
    polyak_update(target_mixer.parameters(), mixer.parameters(), tau)
    for a in AGENTS:
        polyak_update(target_q_nets[a].parameters(), q_nets[a].parameters(), tau)

    # === Diagnostics ===
    with torch.no_grad():
        mean_q_cop = float((q_chosen_per_agent[0] * mask).sum().item() / mask.sum().clamp(min=1.0))
        mean_q_thief = float((q_chosen_per_agent[1] * mask).sum().item() / mask.sum().clamp(min=1.0))
        target_after = torch.cat(
            [p.data.flatten() for p in target_mixer.parameters()]
            + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
        )
        drift = float((target_after - target_before).abs().mean().item())
    return QmixUpdateDiagnostic(
        critic_loss=float(loss.item()),
        mean_q_cop=mean_q_cop,
        mean_q_thief=mean_q_thief,
        target_drift=drift,
    )
