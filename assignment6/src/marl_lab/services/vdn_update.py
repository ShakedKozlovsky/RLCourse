"""VDN update step — wraps apply_qmix_update with a VDNMixer.

Q_tot = Σ Qᵢ (additive). The math is identical to QMIX except the mixer is
stateless. We delegate to apply_qmix_update which is already mixer-agnostic
(it only relies on ``mixer(q_per_agent, global_state)``)."""

from __future__ import annotations

import torch

from marl_lab.memory.centralised_buffer import BatchTensors
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.vdn_mixer import VDNMixer
from marl_lab.services.qmix_update import QmixUpdateDiagnostic, apply_qmix_update


def apply_vdn_update(
    q_nets: dict[str, QPerAgent],
    target_q_nets: dict[str, QPerAgent],
    mixer: VDNMixer,
    target_mixer: VDNMixer,
    batch: BatchTensors,
    gamma: float,
    tau: float,
    critic_opt: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    device: torch.device | None = None,
) -> QmixUpdateDiagnostic:
    """One full VDN update step — uses the same kernel as QMIX."""
    # mypy: VDNMixer is duck-typed as QMIXMixer for apply_qmix_update; it has
    # the same forward(q, state) signature.
    return apply_qmix_update(
        q_nets=q_nets, target_q_nets=target_q_nets,
        mixer=mixer, target_mixer=target_mixer,  # type: ignore[arg-type]
        batch=batch, gamma=gamma, tau=tau, critic_opt=critic_opt,
        max_grad_norm=max_grad_norm, device=device,
    )
