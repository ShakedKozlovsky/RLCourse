"""Construction helpers for MarlTrainer — extracted to keep marl_trainer.py lean."""

from __future__ import annotations

import copy

import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv
from marl_lab.model.qmix_mixer import QMIXMixer
from marl_lab.model.qplex_mixer import QPLEXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.vdn_mixer import VDNMixer

AGENTS = ("cop", "thief")


def build_q_nets(env: DecPomdpEnv, cfg, device: torch.device,
                  n_actions: int) -> tuple[dict, dict]:
    """Build per-agent live + target Q-nets. Targets have requires_grad=False."""
    q_nets = {
        a: QPerAgent(obs_dim=env.obs_dim, n_actions=n_actions,
                      hidden_sizes=cfg.hidden_sizes,
                      gru_hidden_size=cfg.gru_hidden_size).to(device)
        for a in AGENTS
    }
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    return q_nets, target_q_nets


def build_mixer_pair(cfg, state_dim: int, device: torch.device):
    """Build live + target mixer per algorithm. Returns (mixer, target_mixer)
    or (None, None) for IQL."""
    if cfg.algo == "qmix":
        mixer = QMIXMixer(n_agents=2, state_dim=state_dim,
                           embed_dim=cfg.embed_dim,
                           hyper_hidden=cfg.hyper_hidden).to(device)
        target_mixer = copy.deepcopy(mixer)
        for p in target_mixer.parameters():
            p.requires_grad = False
        return mixer, target_mixer
    if cfg.algo == "vdn":
        mixer = VDNMixer(n_agents=2).to(device)
        return mixer, copy.deepcopy(mixer)
    if cfg.algo == "qplex":
        mixer = QPLEXMixer(n_agents=2, state_dim=state_dim,
                            hyper_hidden=cfg.hyper_hidden).to(device)
        target_mixer = copy.deepcopy(mixer)
        for p in target_mixer.parameters():
            p.requires_grad = False
        return mixer, target_mixer
    if cfg.algo == "iql":
        return None, None
    raise ValueError(f"unknown algo: {cfg.algo!r}")


def build_optimisers(cfg, q_nets: dict, mixer):
    """Per-algorithm optimiser setup.

    QMIX/VDN: ONE Adam over (all Q-nets ∪ mixer).
    IQL: TWO Adam (one per agent's Q-net)."""
    if cfg.algo == "iql":
        return {a: torch.optim.Adam(q_nets[a].parameters(), lr=cfg.lr)
                for a in AGENTS}
    params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        params.extend(q_nets[a].parameters())
    params.extend(mixer.parameters())
    return torch.optim.Adam(params, lr=cfg.lr)
