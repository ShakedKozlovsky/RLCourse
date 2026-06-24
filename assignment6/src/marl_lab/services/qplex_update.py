"""QPLEX update step — analogous to QMIX update but with dueling decomposition.

Compared to qmix_update.py the key difference is feeding the mixer
*both* the chosen-action Q values AND the max-action Q values (V_i) so the
mixer can compute A_i = Q_i - V_i internally."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.model.qplex_mixer import QPLEXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.soft_update import polyak_update

AGENTS = ("cop", "thief")


@dataclass(frozen=True)
class QplexUpdateDiagnostic:
    """Per-update logging tuple for QPLEX (mirrors QmixUpdateDiagnostic)."""
    critic_loss: float
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


def apply_qplex_update(
    q_nets: dict[str, QPerAgent],
    target_q_nets: dict[str, QPerAgent],
    mixer: QPLEXMixer,
    target_mixer: QPLEXMixer,
    batch: BatchTensors,
    gamma: float,
    tau: float,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> QplexUpdateDiagnostic:
    """One full QPLEX update step (dueling decomposition).

    Q_tot = V_tot(s) + Σ_i λ_i(s) · (Q_i(τ_i, a_i) − max_a Q_i(τ_i, a))."""
    device = device or torch.device("cpu")
    b = _batch_to_tensors(batch, device)
    target_before = torch.cat(
        [p.data.flatten() for p in target_mixer.parameters()]
        + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
    ).clone()

    # Live nets: per-agent Q_seq, gather chosen action's Q + take max for V
    q_chosen_list: list[torch.Tensor] = []
    v_max_list: list[torch.Tensor] = []
    for a in AGENTS:
        q_seq, _ = q_nets[a](b["obs"][a], hidden=None)        # (B, T, n_actions)
        a_idx = b["action"][a].unsqueeze(-1)                   # (B, T, 1)
        q_chosen = q_seq.gather(dim=-1, index=a_idx).squeeze(-1)
        v_max = q_seq.max(dim=-1).values                       # (B, T)
        q_chosen_list.append(q_chosen)
        v_max_list.append(v_max)
    q_chosen_stack = torch.stack(q_chosen_list, dim=-1)        # (B, T, n_agents)
    v_max_stack = torch.stack(v_max_list, dim=-1)              # (B, T, n_agents)
    q_tot = mixer(q_chosen_stack, v_max_stack, b["state"])

    # Target side: max-action Q (which is its own V) → q == v in the target call
    with torch.no_grad():
        v_next_list: list[torch.Tensor] = []
        for a in AGENTS:
            q_next_seq, _ = target_q_nets[a](b["next_obs"][a], hidden=None)
            v_next = q_next_seq.max(dim=-1).values
            v_next_list.append(v_next)
        v_next_stack = torch.stack(v_next_list, dim=-1)
        # At argmax, Q == V so A = 0 → Q_tot_target = V_tot_target only.
        # Equivalently, pass v_next_stack as both q and v so adv == 0.
        q_tot_next = target_mixer(v_next_stack, v_next_stack, b["next_state"])
        joint_reward = (b["reward"]["cop"] + b["reward"]["thief"]) * 0.5
        y = joint_reward + gamma * (1.0 - b["done"]) * q_tot_next

    mask = b["mask"]
    loss = (((q_tot - y) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    critic_opt.zero_grad()
    loss.backward()
    all_params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        all_params.extend(q_nets[a].parameters())
    all_params.extend(mixer.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
    critic_opt.step()

    polyak_update(target_mixer.parameters(), mixer.parameters(), tau)
    for a in AGENTS:
        polyak_update(target_q_nets[a].parameters(), q_nets[a].parameters(), tau)

    with torch.no_grad():
        mean_q_cop = float((q_chosen_list[0] * mask).sum().item() / mask.sum().clamp(min=1.0))
        mean_q_thief = float((q_chosen_list[1] * mask).sum().item() / mask.sum().clamp(min=1.0))
        target_after = torch.cat(
            [p.data.flatten() for p in target_mixer.parameters()]
            + [p.data.flatten() for a in AGENTS for p in target_q_nets[a].parameters()]
        )
        drift = float((target_after - target_before).abs().mean().item())
    return QplexUpdateDiagnostic(
        critic_loss=float(loss.item()),
        mean_q_cop=mean_q_cop,
        mean_q_thief=mean_q_thief,
        target_drift=drift,
    )
