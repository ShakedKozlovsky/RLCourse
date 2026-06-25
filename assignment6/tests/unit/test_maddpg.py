"""MADDPG-discrete tests — per-agent centralised critic + per-agent reward."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.model.maddpg_critic import MADDPGCritic, one_hot_joint_action
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.maddpg_update import apply_maddpg_update
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig
from marl_lab.shared.types import EpisodeSequence, Transition

AGENTS = ("cop", "thief")


# ----- MADDPGCritic -----

def test_maddpg_critic_validates_n_actions() -> None:
    with pytest.raises(ValueError):
        MADDPGCritic(state_dim=10, n_actions_per_agent=(0, 5))


def test_maddpg_critic_output_shape() -> None:
    c = MADDPGCritic(state_dim=10, n_actions_per_agent=(6, 5),
                       hidden_sizes=(32,))
    s = torch.randn(4, 5, 10)
    a = torch.zeros(4, 5, 11)   # 6+5 one-hot
    out = c(s, a)
    assert out.shape == (4, 5)


def test_maddpg_critic_rejects_wrong_state_dim() -> None:
    c = MADDPGCritic(state_dim=10, n_actions_per_agent=(6, 5))
    with pytest.raises(ValueError):
        c(torch.zeros(2, 9), torch.zeros(2, 11))


def test_maddpg_critic_rejects_wrong_action_dim() -> None:
    c = MADDPGCritic(state_dim=10, n_actions_per_agent=(6, 5))
    with pytest.raises(ValueError):
        c(torch.zeros(2, 10), torch.zeros(2, 10))   # should be 11


def test_one_hot_joint_action_concat_order() -> None:
    actions = {
        "cop": torch.tensor([0, 1, 2]),
        "thief": torch.tensor([3, 4, 0]),
    }
    n_per = {"cop": 6, "thief": 5}
    oh = one_hot_joint_action(actions, n_per)
    assert oh.shape == (3, 11)        # 6 + 5
    # First 6 entries of row 0 must be one-hot at idx 0
    assert oh[0, 0] == 1.0 and oh[0, 1:6].sum() == 0
    # Next 5 entries of row 0 must be one-hot at idx 3
    assert oh[0, 6 + 3] == 1.0 and oh[0, 6:6 + 3].sum() == 0


def test_one_hot_joint_action_dtype_is_float32() -> None:
    actions = {"cop": torch.tensor([0]), "thief": torch.tensor([0])}
    oh = one_hot_joint_action(actions, {"cop": 6, "thief": 5})
    assert oh.dtype == torch.float32


# ----- MADDPG update step -----

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


def _kit() -> dict:
    """Build the full MADDPG training kit (Q-nets + critics + buffer + opt)."""
    obs_dim, state_dim = 8, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(16,),
                              gru_hidden_size=8) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    critics = {a: MADDPGCritic(state_dim=state_dim,
                                  n_actions_per_agent=(6, 5),
                                  hidden_sizes=(16,)) for a in AGENTS}
    target_critics = {a: copy.deepcopy(critics[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_critics[a].parameters():
            p.requires_grad = False
    params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        params.extend(q_nets[a].parameters())
        params.extend(critics[a].parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    buf = CentralisedReplayBuffer(capacity=8, max_seq_len=6, state_dim=state_dim,
                                    obs_dim=obs_dim,
                                    n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    for _ in range(8):
        buf.push(_ep(6, obs_dim, state_dim))
    return {
        "q_nets": q_nets, "target_q_nets": target_q_nets,
        "critics": critics, "target_critics": target_critics,
        "buf": buf, "opt": opt,
    }


def test_maddpg_update_returns_finite_per_agent_losses() -> None:
    k = _kit()
    diag = apply_maddpg_update(
        q_nets=k["q_nets"], target_q_nets=k["target_q_nets"],
        critics=k["critics"], target_critics=k["target_critics"],
        batch=k["buf"].sample(4), gamma=0.99, tau=0.005, critic_opt=k["opt"],
    )
    assert np.isfinite(diag.critic_loss_cop)
    assert np.isfinite(diag.critic_loss_thief)


def test_maddpg_update_changes_all_weights() -> None:
    k = _kit()
    q_before = next(k["q_nets"]["cop"].parameters()).clone()
    c_before = next(k["critics"]["cop"].parameters()).clone()
    apply_maddpg_update(
        q_nets=k["q_nets"], target_q_nets=k["target_q_nets"],
        critics=k["critics"], target_critics=k["target_critics"],
        batch=k["buf"].sample(4), gamma=0.99, tau=0.005, critic_opt=k["opt"],
    )
    assert (q_before - next(k["q_nets"]["cop"].parameters())).abs().max() > 1e-7
    assert (c_before - next(k["critics"]["cop"].parameters())).abs().max() > 1e-7


def test_maddpg_update_target_drift_positive() -> None:
    k = _kit()
    diag = apply_maddpg_update(
        q_nets=k["q_nets"], target_q_nets=k["target_q_nets"],
        critics=k["critics"], target_critics=k["target_critics"],
        batch=k["buf"].sample(4), gamma=0.99, tau=0.005, critic_opt=k["opt"],
    )
    assert diag.target_drift > 0.0


def test_maddpg_update_no_grad_on_targets() -> None:
    k = _kit()
    apply_maddpg_update(
        q_nets=k["q_nets"], target_q_nets=k["target_q_nets"],
        critics=k["critics"], target_critics=k["target_critics"],
        batch=k["buf"].sample(4), gamma=0.99, tau=0.005, critic_opt=k["opt"],
    )
    for a in AGENTS:
        for p in k["target_q_nets"][a].parameters():
            assert p.grad is None
        for p in k["target_critics"][a].parameters():
            assert p.grad is None


def test_maddpg_per_agent_reward_actually_distinguishes_agents() -> None:
    """The cop's critic loss should differ from the thief's (we use per-agent
    reward, not the averaged QMIX-style joint reward)."""
    k = _kit()
    diag = apply_maddpg_update(
        q_nets=k["q_nets"], target_q_nets=k["target_q_nets"],
        critics=k["critics"], target_critics=k["target_critics"],
        batch=k["buf"].sample(4), gamma=0.99, tau=0.005, critic_opt=k["opt"],
    )
    # With independent random reward streams, the per-agent losses will
    # almost certainly differ — confirms POSG-fidelity, not Dec-POMDP averaging.
    assert diag.critic_loss_cop != diag.critic_loss_thief


# ----- End-to-end trainer integration -----

def test_maddpg_in_trainer_end_to_end() -> None:
    """MarlTrainer with algo='maddpg' completes 4 episodes including
    post-warmup learning."""
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(4, 4), max_moves=6, max_barriers=2,
                          enable_barriers=False, observation_radius=1),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )
    env.reset(seed=0)
    cfg = TrainerConfig(algo="maddpg", batch_size=4, buffer_capacity=16,
                          warmup_episodes=2, max_seq_len=6, embed_dim=8,
                          hyper_hidden=16, gru_hidden_size=8, hidden_sizes=(16,))
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=20)
    trainer = MarlTrainer(env, cfg, sched, rng=np.random.default_rng(0))
    history = trainer.train(n_episodes=4)
    assert len(history) == 4
    # Post-warmup, at least one step has finite non-zero critic_loss
    assert any(h.critic_loss != 0.0 for h in history[-2:])
    # MADDPG owns critics dict
    assert "cop" in trainer.critics
    assert "thief" in trainer.critics


def test_maddpg_trainer_critics_have_centralised_state_dim() -> None:
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(5, 5), max_moves=6, max_barriers=2,
                          enable_barriers=False, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )
    env.reset(seed=0)
    cfg = TrainerConfig(algo="maddpg", batch_size=4, buffer_capacity=16,
                          warmup_episodes=2, max_seq_len=6, embed_dim=8,
                          hyper_hidden=16, gru_hidden_size=8, hidden_sizes=(16,))
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=20)
    trainer = MarlTrainer(env, cfg, sched, rng=np.random.default_rng(0))
    expected_state_dim = trainer.env.global_state().shape[0]
    assert trainer.critics["cop"].state_dim == expected_state_dim
    assert trainer.critics["thief"].state_dim == expected_state_dim
