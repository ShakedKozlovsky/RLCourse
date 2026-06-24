"""Layer 11 — VDN + IQL update tests."""

from __future__ import annotations

import copy

import numpy as np
import torch

from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.model.vdn_mixer import VDNMixer
from marl_lab.services.iql_update import apply_iql_update
from marl_lab.services.vdn_update import apply_vdn_update
from marl_lab.shared.types import EpisodeSequence, Transition

AGENTS = ("cop", "thief")


def _ep(length: int, obs_dim: int, state_dim: int) -> EpisodeSequence:
    rng = np.random.default_rng(0)
    ep = EpisodeSequence()
    for t in range(length):
        ep.transitions.append(Transition(
            global_state=rng.standard_normal(state_dim).astype(np.float32),
            joint_obs={a: rng.standard_normal(obs_dim).astype(np.float32) for a in AGENTS},
            joint_action={"cop": int(rng.integers(0, 6)), "thief": int(rng.integers(0, 5))},
            joint_reward={a: float(rng.standard_normal()) for a in AGENTS},
            next_global_state=rng.standard_normal(state_dim).astype(np.float32),
            next_joint_obs={a: rng.standard_normal(obs_dim).astype(np.float32) for a in AGENTS},
            done=(t == length - 1),
        ))
    return ep


def _fill_buffer(obs_dim: int, state_dim: int) -> CentralisedReplayBuffer:
    buf = CentralisedReplayBuffer(capacity=8, max_seq_len=12, state_dim=state_dim,
                                    obs_dim=obs_dim,
                                    n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    for _ in range(8):
        buf.push(_ep(10, obs_dim, state_dim))
    return buf


# ----- VDN -----

def test_vdn_update_returns_finite_loss() -> None:
    obs_dim, state_dim = 10, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(32,),
                              gru_hidden_size=16) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    mixer = VDNMixer(n_agents=2)
    target_mixer = copy.deepcopy(mixer)
    params = [p for a in AGENTS for p in q_nets[a].parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    buf = _fill_buffer(obs_dim, state_dim)
    batch = buf.sample(4)
    diag = apply_vdn_update(q_nets, target_q_nets, mixer, target_mixer,
                              batch, gamma=0.99, tau=0.005, critic_opt=opt)
    assert np.isfinite(diag.critic_loss)
    assert diag.critic_loss >= 0.0


def test_vdn_update_changes_q_weights() -> None:
    obs_dim, state_dim = 10, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(32,),
                              gru_hidden_size=16) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    mixer = VDNMixer(n_agents=2)
    target_mixer = copy.deepcopy(mixer)
    params = [p for a in AGENTS for p in q_nets[a].parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    buf = _fill_buffer(obs_dim, state_dim)
    batch = buf.sample(4)
    before = next(q_nets["cop"].parameters()).clone()
    apply_vdn_update(q_nets, target_q_nets, mixer, target_mixer,
                     batch, gamma=0.99, tau=0.005, critic_opt=opt)
    after = next(q_nets["cop"].parameters())
    assert (before - after).abs().max() > 1e-7


# ----- IQL -----

def test_iql_update_returns_finite_per_agent_losses() -> None:
    obs_dim, state_dim = 10, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(32,),
                              gru_hidden_size=16) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    opts = {a: torch.optim.Adam(q_nets[a].parameters(), lr=1e-3) for a in AGENTS}
    buf = _fill_buffer(obs_dim, state_dim)
    batch = buf.sample(4)
    diag = apply_iql_update(q_nets, target_q_nets, batch,
                            gamma=0.99, tau=0.005, critic_opts=opts)
    assert np.isfinite(diag.critic_loss_cop)
    assert np.isfinite(diag.critic_loss_thief)


def test_iql_update_changes_each_q_independently() -> None:
    obs_dim, state_dim = 10, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(32,),
                              gru_hidden_size=16) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    opts = {a: torch.optim.Adam(q_nets[a].parameters(), lr=1e-3) for a in AGENTS}
    buf = _fill_buffer(obs_dim, state_dim)
    batch = buf.sample(4)
    cop_before = next(q_nets["cop"].parameters()).clone()
    thief_before = next(q_nets["thief"].parameters()).clone()
    apply_iql_update(q_nets, target_q_nets, batch,
                     gamma=0.99, tau=0.005, critic_opts=opts)
    assert (cop_before - next(q_nets["cop"].parameters())).abs().max() > 1e-7
    assert (thief_before - next(q_nets["thief"].parameters())).abs().max() > 1e-7


def test_iql_update_no_grad_on_targets() -> None:
    obs_dim, state_dim = 10, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(32,),
                              gru_hidden_size=16) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    opts = {a: torch.optim.Adam(q_nets[a].parameters(), lr=1e-3) for a in AGENTS}
    buf = _fill_buffer(obs_dim, state_dim)
    batch = buf.sample(4)
    apply_iql_update(q_nets, target_q_nets, batch,
                     gamma=0.99, tau=0.005, critic_opts=opts)
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            assert p.grad is None
