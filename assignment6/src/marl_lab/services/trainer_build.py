"""Construction helpers for MarlTrainer — extracted to keep marl_trainer.py lean."""

from __future__ import annotations

import copy

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.model.maddpg_critic import MADDPGCritic
from marl_lab.model.qmix_mixer import QMIXMixer
from marl_lab.model.qplex_mixer import QPLEXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.vdn_mixer import VDNMixer

AGENTS = ("cop", "thief")
N_PER_AGENT = {"cop": 6, "thief": 5}


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
    if cfg.algo in ("iql", "maddpg"):
        # MADDPG uses per-agent critics (built separately via build_maddpg_critics)
        return None, None
    raise ValueError(f"unknown algo: {cfg.algo!r}")


def build_maddpg_critics(cfg, state_dim: int, device: torch.device,
                          ) -> tuple[dict, dict]:
    """Build per-agent centralised critics + frozen targets for MADDPG."""
    critics = {
        a: MADDPGCritic(
            state_dim=state_dim,
            n_actions_per_agent=(N_PER_AGENT["cop"], N_PER_AGENT["thief"]),
            hidden_sizes=tuple(cfg.hidden_sizes),
        ).to(device)
        for a in AGENTS
    }
    target_critics = {a: copy.deepcopy(critics[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_critics[a].parameters():
            p.requires_grad = False
    return critics, target_critics


def build_optimisers(cfg, q_nets: dict, mixer, critics: dict | None = None):
    """Per-algorithm optimiser setup.

    QMIX/VDN/QPLEX: ONE Adam over (all Q-nets ∪ mixer).
    IQL: TWO Adam (one per agent's Q-net).
    MADDPG: ONE Adam over (all Q-nets ∪ all critics)."""
    if cfg.algo == "iql":
        return {a: torch.optim.Adam(q_nets[a].parameters(), lr=cfg.lr)
                for a in AGENTS}
    if cfg.algo == "maddpg":
        if critics is None:
            raise ValueError("maddpg algo requires critics= passed in")
        params: list[torch.nn.Parameter] = []
        for a in AGENTS:
            params.extend(q_nets[a].parameters())
            params.extend(critics[a].parameters())
        return torch.optim.Adam(params, lr=cfg.lr)
    params2: list[torch.nn.Parameter] = []
    for a in AGENTS:
        params2.extend(q_nets[a].parameters())
    params2.extend(mixer.parameters())
    return torch.optim.Adam(params2, lr=cfg.lr)


def rebuild_env_and_mixer_for_grid(
    *, old_env: DecPomdpEnv, grid_size: tuple[int, int],
    cfg, q_nets: dict, device: torch.device, rng: np.random.Generator,
):
    """Build (env, mixer, target_mixer, opts, buffer) for a new grid size.

    Curriculum semantics — Q-nets are NOT rebuilt (their obs_dim depends on
    observation_radius only). Returns the new components as a tuple."""
    env_cfg = EnvConfig(
        grid_size=grid_size,
        max_moves=max(8, grid_size[0] * grid_size[1]),
        max_barriers=old_env.env_cfg.max_barriers,
        enable_barriers=old_env.env_cfg.enable_barriers,
        observation_radius=old_env.env_cfg.observation_radius,
    )
    env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=RewardConfig(), rng=rng)
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    state_dim = env.global_state().shape[0]
    mixer, target_mixer = build_mixer_pair(cfg, state_dim, device)
    opts = build_optimisers(cfg, q_nets, mixer)
    buffer = CentralisedReplayBuffer(
        capacity=cfg.buffer_capacity, max_seq_len=env_cfg.max_moves,
        state_dim=state_dim, obs_dim=env.obs_dim,
        n_actions_per_agent={"cop": 6, "thief": 5},
        rng=rng,
    )
    return env, mixer, target_mixer, opts, buffer
