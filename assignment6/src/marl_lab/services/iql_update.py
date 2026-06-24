"""IQL — Independent Q-Learning baseline (Foerster et al. 2018).

Each agent runs DQN on its OWN reward stream — no centralised critic, no
mixer. The point is to demonstrate the non-stationarity problem the CTDE
methods (VDN/QMIX) fix: from each agent's perspective the environment
changes as the OTHER agent's policy changes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.soft_update import polyak_update

AGENTS = ("cop", "thief")


@dataclass(frozen=True)
class IqlUpdateDiagnostic:
    """Per-update logging tuple for IQL (per-agent losses)."""
    critic_loss_cop: float
    critic_loss_thief: float
    mean_q_cop: float
    mean_q_thief: float


def apply_iql_update(
    q_nets: dict[str, QPerAgent],
    target_q_nets: dict[str, QPerAgent],
    batch: BatchTensors,
    gamma: float,
    tau: float,
    critic_opts: dict[str, torch.optim.Optimizer],
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> IqlUpdateDiagnostic:
    """One IQL update step — independent DQN per agent.

    Each agent uses its OWN reward and its OWN target net; there's no global
    state and no mixer. Shows the non-stationarity baseline."""
    device = device or torch.device("cpu")
    diag: dict[str, float] = {}
    mean_q: dict[str, float] = {}
    for a in AGENTS:
        obs = torch.as_tensor(batch.obs_seq[a], device=device)
        next_obs = torch.as_tensor(batch.next_obs_seq[a], device=device)
        action = torch.as_tensor(batch.action_seq[a], device=device)
        reward = torch.as_tensor(batch.reward_seq[a], device=device)
        done = torch.as_tensor(batch.done_seq, device=device)
        mask = torch.as_tensor(batch.mask, device=device)

        # Forward live net
        q_seq, _ = q_nets[a](obs, hidden=None)
        q_chosen = q_seq.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        # Target
        with torch.no_grad():
            q_next, _ = target_q_nets[a](next_obs, hidden=None)
            q_next_max = q_next.max(dim=-1).values
            y = reward + gamma * (1.0 - done) * q_next_max
        # Masked MSE
        td_error = (q_chosen - y) ** 2
        loss = (td_error * mask).sum() / mask.sum().clamp(min=1.0)

        critic_opts[a].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_nets[a].parameters(), max_grad_norm)
        critic_opts[a].step()

        # Polyak
        polyak_update(target_q_nets[a].parameters(), q_nets[a].parameters(), tau)
        diag[a] = float(loss.item())
        with torch.no_grad():
            mean_q[a] = float((q_chosen * mask).sum().item() / mask.sum().clamp(min=1.0))

    return IqlUpdateDiagnostic(
        critic_loss_cop=diag["cop"],
        critic_loss_thief=diag["thief"],
        mean_q_cop=mean_q["cop"],
        mean_q_thief=mean_q["thief"],
    )
