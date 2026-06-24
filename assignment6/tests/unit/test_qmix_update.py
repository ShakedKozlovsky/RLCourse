"""Layer 10 — QMIX update gradient + math battery (the headline TDD pair)."""

from __future__ import annotations

import copy

import numpy as np
import torch

from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.model.qmix_mixer import QMIXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.services.qmix_update import apply_qmix_update
from marl_lab.shared.types import EpisodeSequence, Transition

AGENTS = ("cop", "thief")


def _dummy_episode(length: int, obs_dim: int, state_dim: int,
                   n_cop_actions: int = 6, n_thief_actions: int = 5) -> EpisodeSequence:
    """Build a synthetic episode of the requested length for buffer-fill tests."""
    ep = EpisodeSequence()
    rng = np.random.default_rng(0)
    for t in range(length):
        ep.transitions.append(Transition(
            global_state=rng.standard_normal(state_dim).astype(np.float32),
            joint_obs={"cop": rng.standard_normal(obs_dim).astype(np.float32),
                       "thief": rng.standard_normal(obs_dim).astype(np.float32)},
            joint_action={"cop": int(rng.integers(0, n_cop_actions)),
                          "thief": int(rng.integers(0, n_thief_actions))},
            joint_reward={"cop": float(rng.standard_normal()),
                          "thief": float(rng.standard_normal())},
            next_global_state=rng.standard_normal(state_dim).astype(np.float32),
            next_joint_obs={"cop": rng.standard_normal(obs_dim).astype(np.float32),
                            "thief": rng.standard_normal(obs_dim).astype(np.float32)},
            done=(t == length - 1),
        ))
    return ep


def _setup() -> dict:
    """Build the full QMIX kit: Q-nets, targets, mixer, target mixer, optim, batch."""
    obs_dim, state_dim = 10, 6
    n_cop, n_thief = 6, 5
    # Use the LARGER action space as canonical Q output (n_actions = 6); thief
    # will only-ever pick from [0..n_thief-1] in the dataset, but the net's
    # output dim is identical, simplifying batching.
    q_nets = {
        "cop": QPerAgent(obs_dim=obs_dim, n_actions=n_cop, hidden_sizes=(32,), gru_hidden_size=16),
        "thief": QPerAgent(obs_dim=obs_dim, n_actions=n_cop, hidden_sizes=(32,), gru_hidden_size=16),
    }
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    mixer = QMIXMixer(n_agents=2, state_dim=state_dim, embed_dim=8, hyper_hidden=16)
    target_mixer = copy.deepcopy(mixer)
    for p in target_mixer.parameters():
        p.requires_grad = False
    all_params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        all_params.extend(q_nets[a].parameters())
    all_params.extend(mixer.parameters())
    critic_opt = torch.optim.Adam(all_params, lr=1e-3)
    # Fill a tiny buffer with a few episodes
    buf = CentralisedReplayBuffer(capacity=8, max_seq_len=12, state_dim=state_dim,
                                    obs_dim=obs_dim,
                                    n_actions_per_agent={"cop": n_cop, "thief": n_thief},
                                    rng=np.random.default_rng(0))
    for _ in range(8):
        buf.push(_dummy_episode(length=10, obs_dim=obs_dim, state_dim=state_dim,
                                n_cop_actions=n_cop, n_thief_actions=n_thief))
    return {
        "q_nets": q_nets, "target_q_nets": target_q_nets,
        "mixer": mixer, "target_mixer": target_mixer,
        "critic_opt": critic_opt, "buf": buf, "obs_dim": obs_dim,
        "state_dim": state_dim,
    }


def test_qmix_update_returns_diagnostic() -> None:
    kit = _setup()
    batch = kit["buf"].sample(4)
    diag = apply_qmix_update(
        q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
        mixer=kit["mixer"], target_mixer=kit["target_mixer"],
        batch=batch, gamma=0.99, tau=0.005, critic_opt=kit["critic_opt"],
    )
    assert isinstance(diag.critic_loss, float)
    assert isinstance(diag.target_drift, float)
    assert diag.critic_loss >= 0.0


def test_qmix_update_changes_live_weights() -> None:
    kit = _setup()
    batch = kit["buf"].sample(4)
    cop_q_before = next(kit["q_nets"]["cop"].parameters()).clone()
    mixer_before = next(kit["mixer"].parameters()).clone()
    apply_qmix_update(
        q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
        mixer=kit["mixer"], target_mixer=kit["target_mixer"],
        batch=batch, gamma=0.99, tau=0.005, critic_opt=kit["critic_opt"],
    )
    cop_q_after = next(kit["q_nets"]["cop"].parameters())
    mixer_after = next(kit["mixer"].parameters())
    assert (cop_q_before - cop_q_after).abs().max() > 1e-7
    assert (mixer_before - mixer_after).abs().max() > 1e-7


def test_qmix_update_no_grad_on_targets() -> None:
    """After backward + step, target params have no .grad (they're frozen)."""
    kit = _setup()
    batch = kit["buf"].sample(4)
    apply_qmix_update(
        q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
        mixer=kit["mixer"], target_mixer=kit["target_mixer"],
        batch=batch, gamma=0.99, tau=0.005, critic_opt=kit["critic_opt"],
    )
    for a in AGENTS:
        for p in kit["target_q_nets"][a].parameters():
            assert p.grad is None
    for p in kit["target_mixer"].parameters():
        assert p.grad is None


def test_qmix_target_drift_positive_at_nonzero_tau() -> None:
    """target_drift should be > 0 when τ = 0.005 (Polyak actually moved)."""
    kit = _setup()
    batch = kit["buf"].sample(4)
    diag = apply_qmix_update(
        q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
        mixer=kit["mixer"], target_mixer=kit["target_mixer"],
        batch=batch, gamma=0.99, tau=0.005, critic_opt=kit["critic_opt"],
    )
    assert diag.target_drift > 0.0


def test_qmix_tau_zero_freezes_targets() -> None:
    """τ = 0 ⇒ target weights unchanged after update."""
    kit = _setup()
    batch = kit["buf"].sample(4)
    tmixer_before = next(kit["target_mixer"].parameters()).clone()
    apply_qmix_update(
        q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
        mixer=kit["mixer"], target_mixer=kit["target_mixer"],
        batch=batch, gamma=0.99, tau=0.0, critic_opt=kit["critic_opt"],
    )
    tmixer_after = next(kit["target_mixer"].parameters())
    torch.testing.assert_close(tmixer_before, tmixer_after)


def test_qmix_loss_finite_on_random_data() -> None:
    """Loss is finite over 5 consecutive updates with random data."""
    kit = _setup()
    for _ in range(5):
        batch = kit["buf"].sample(4)
        diag = apply_qmix_update(
            q_nets=kit["q_nets"], target_q_nets=kit["target_q_nets"],
            mixer=kit["mixer"], target_mixer=kit["target_mixer"],
            batch=batch, gamma=0.99, tau=0.005, critic_opt=kit["critic_opt"],
        )
        assert np.isfinite(diag.critic_loss)
        assert np.isfinite(diag.mean_q_cop)
        assert np.isfinite(diag.mean_q_thief)
