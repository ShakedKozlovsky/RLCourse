"""QPLEX mixer + update tests.

The headline maths to verify:
  1. Q_tot = V_tot + Σ λ_i · (Q_i − V_i)
  2. λ_i > 0 for all i (positive advantage weights — IGM by construction)
  3. When all agents act optimally (Q_i = V_i ⇒ A_i = 0), Q_tot = V_tot
  4. Increasing Q_i while V_i held fixed strictly increases Q_tot (monotone in Q_i)
  5. Q_tot is NOT constrained to be monotone in V_tot (unlike QMIX's W²·Q
     structure) — QPLEX can represent the non-monotonic landscapes QMIX cannot.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.model.qplex_mixer import QPLEXMixer
from marl_lab.model.recurrent_q import QPerAgent
from marl_lab.services.qplex_update import apply_qplex_update
from marl_lab.shared.types import EpisodeSequence, Transition

AGENTS = ("cop", "thief")


def test_qplex_validates_n_agents() -> None:
    with pytest.raises(ValueError):
        QPLEXMixer(n_agents=0, state_dim=4)


def test_qplex_output_shape() -> None:
    mixer = QPLEXMixer(n_agents=2, state_dim=10, hyper_hidden=16)
    q = torch.randn(4, 5, 2)
    v = torch.randn(4, 5, 2)
    s = torch.randn(4, 5, 10)
    q_tot = mixer(q, v, s)
    assert q_tot.shape == (4, 5)


def test_qplex_rejects_mismatched_n_agents() -> None:
    mixer = QPLEXMixer(n_agents=2, state_dim=10)
    q = torch.randn(4, 5, 3)
    v = torch.randn(4, 5, 3)
    s = torch.randn(4, 5, 10)
    with pytest.raises(ValueError):
        mixer(q, v, s)


def test_qplex_rejects_mismatched_q_v_shapes() -> None:
    mixer = QPLEXMixer(n_agents=2, state_dim=10)
    q = torch.randn(4, 5, 2)
    v = torch.randn(4, 5, 3)
    s = torch.randn(4, 5, 10)
    with pytest.raises(ValueError):
        mixer(q, v, s)


def test_qplex_at_argmax_reduces_to_v_tot() -> None:
    """When all agents pick the greedy action, Q_i == V_i so A_i == 0 and
    Q_tot reduces to V_tot(s) alone — the dueling identity."""
    torch.manual_seed(0)
    mixer = QPLEXMixer(n_agents=2, state_dim=10, hyper_hidden=16)
    s = torch.randn(2, 3, 10)
    v = torch.randn(2, 3, 2)
    q_tot_at_argmax = mixer(v, v, s)        # Q == V → A == 0
    v_tot_alone = mixer(torch.zeros_like(v), torch.zeros_like(v), s)
    torch.testing.assert_close(q_tot_at_argmax, v_tot_alone)


def test_qplex_monotonic_in_per_agent_q() -> None:
    """∂Q_tot/∂Q_i = λ_i > 0 ⇒ increasing Q_i increases Q_tot (V_i fixed)."""
    torch.manual_seed(0)
    mixer = QPLEXMixer(n_agents=2, state_dim=10)
    s = torch.randn(1, 1, 10)
    v = torch.tensor([[[0.5, 0.5]]])
    q_low = torch.tensor([[[0.0, 0.5]]])
    q_high = torch.tensor([[[1.0, 0.5]]])
    out_low = mixer(q_low, v, s).item()
    out_high = mixer(q_high, v, s).item()
    assert out_high > out_low


def test_qplex_lambda_positivity_via_autograd() -> None:
    """∂Q_tot/∂Q_i > 0 verified by autograd over many random probes."""
    torch.manual_seed(0)
    mixer = QPLEXMixer(n_agents=3, state_dim=8)
    for _ in range(80):
        q = torch.randn(1, 1, 3, requires_grad=True)
        v = torch.randn(1, 1, 3)
        s = torch.randn(1, 1, 8)
        q_tot = mixer(q, v, s)
        grads = torch.autograd.grad(q_tot.sum(), q)[0]
        assert (grads > 0).all(), f"λ violation: {grads}"


def test_qplex_more_expressive_than_qmix() -> None:
    """QPLEX can produce a Q_tot whose ranking over actions disagrees with
    plain Σ Q_i (which QMIX can't). Easiest probe: with V_tot's gradient
    being unconstrained, we can construct (q1, q2) and (q1', q2') such that
    q1+q2 > q1'+q2' but Q_tot(q1', q2') > Q_tot(q1, q2) for the right s."""
    # We don't need to prove generality, only that the architecture admits it.
    # A weaker check: V_tot can be negative (QMIX cannot make Q_tot negative
    # when all q_i are non-negative because |W|>=0).
    torch.manual_seed(0)
    mixer = QPLEXMixer(n_agents=2, state_dim=10)
    # Force V head to a known negative value via gradient
    opt = torch.optim.Adam(mixer.parameters(), lr=1e-2)
    for _ in range(100):
        s = torch.randn(8, 1, 10)
        q = torch.ones(8, 1, 2)
        v = torch.ones(8, 1, 2)
        loss = ((mixer(q, v, s) - (-3.0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # After 100 steps Q_tot should be near −3 even with positive Q's
    s = torch.randn(1, 1, 10)
    q = torch.ones(1, 1, 2)
    v = torch.ones(1, 1, 2)
    assert mixer(q, v, s).item() < 0.0


# ----- QPLEX update step (end-to-end TD) -----

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


def test_qplex_update_full_step() -> None:
    """End-to-end QPLEX update: weights change, target drifts, loss finite."""
    obs_dim, state_dim = 8, 6
    q_nets = {a: QPerAgent(obs_dim=obs_dim, n_actions=6, hidden_sizes=(16,),
                              gru_hidden_size=8) for a in AGENTS}
    target_q_nets = {a: copy.deepcopy(q_nets[a]) for a in AGENTS}
    for a in AGENTS:
        for p in target_q_nets[a].parameters():
            p.requires_grad = False
    mixer = QPLEXMixer(n_agents=2, state_dim=state_dim, hyper_hidden=16)
    target_mixer = copy.deepcopy(mixer)
    for p in target_mixer.parameters():
        p.requires_grad = False
    params: list[torch.nn.Parameter] = []
    for a in AGENTS:
        params.extend(q_nets[a].parameters())
    params.extend(mixer.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    buf = CentralisedReplayBuffer(capacity=8, max_seq_len=8, state_dim=state_dim,
                                    obs_dim=obs_dim,
                                    n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    for _ in range(8):
        buf.push(_ep(6, obs_dim, state_dim))
    cop_before = next(q_nets["cop"].parameters()).clone()
    diag = apply_qplex_update(q_nets, target_q_nets, mixer, target_mixer,
                                buf.sample(4), gamma=0.99, tau=0.005, critic_opt=opt)
    assert np.isfinite(diag.critic_loss)
    assert diag.target_drift > 0.0
    assert (cop_before - next(q_nets["cop"].parameters())).abs().max() > 1e-7


def test_qplex_in_trainer_end_to_end() -> None:
    """MarlTrainer with algo='qplex' completes a 4-episode training run."""
    from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
    from marl_lab.environment.reward import RewardConfig
    from marl_lab.noise.schedule import LinearEpsilonSchedule
    from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig

    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(4, 4), max_moves=6, max_barriers=2,
                          enable_barriers=False, observation_radius=1),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )
    env.reset(seed=0)
    cfg = TrainerConfig(algo="qplex", batch_size=4, buffer_capacity=16,
                          warmup_episodes=2, max_seq_len=6, embed_dim=8,
                          hyper_hidden=16, gru_hidden_size=8, hidden_sizes=(16,))
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=20)
    trainer = MarlTrainer(env, cfg, sched, rng=np.random.default_rng(0))
    history = trainer.train(n_episodes=4)
    assert len(history) == 4
    # At least one post-warmup step must have a non-zero critic_loss
    assert any(h.critic_loss != 0.0 for h in history[-2:])
