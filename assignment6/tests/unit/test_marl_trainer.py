"""Layer 12 — end-to-end MARL trainer smoke tests (all three algos)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig


def _make_env(seed: int = 0) -> DecPomdpEnv:
    return DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(5, 5), max_moves=15, max_barriers=3,
                          enable_barriers=True, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(seed),
    )


def _make_trainer(algo: str, seed: int = 0) -> MarlTrainer:
    env = _make_env(seed)
    env.reset(seed=seed)
    cfg = TrainerConfig(algo=algo, batch_size=4, buffer_capacity=32,
                        warmup_episodes=2, max_seq_len=15, embed_dim=8,
                        hyper_hidden=16, gru_hidden_size=16, hidden_sizes=(32,))
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=100)
    return MarlTrainer(env, cfg, sched, rng=np.random.default_rng(seed))


@pytest.mark.parametrize("algo", ["qmix", "vdn", "iql"])
def test_trainer_collect_episode_returns_nonempty_sequence(algo: str) -> None:
    tr = _make_trainer(algo)
    ep, diag = tr.collect_episode(seed=0)
    assert len(ep) >= 1
    assert diag.episode_steps >= 1
    assert diag.winner in ("cop", "thief")


@pytest.mark.parametrize("algo", ["qmix", "vdn", "iql"])
def test_trainer_train_loop_runs_no_errors(algo: str) -> None:
    """Run 4 episodes — first 2 are warmup, last 2 must train without throwing."""
    tr = _make_trainer(algo)
    history = tr.train(n_episodes=4)
    assert len(history) == 4
    # After warmup at least one step should have a non-zero critic_loss
    assert any(np.isfinite(d.critic_loss) and d.critic_loss != 0.0 for d in history[-2:])


def test_trainer_iql_rejects_invalid_algo() -> None:
    env = _make_env(0)
    env.reset(seed=0)
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=100)
    with pytest.raises(ValueError):
        MarlTrainer(env, TrainerConfig(algo="actor_critic"), sched)


def test_trainer_qmix_changes_q_weights_after_warmup() -> None:
    tr = _make_trainer("qmix")
    before = next(tr.q_nets["cop"].parameters()).clone()
    tr.train(n_episodes=4)
    after = next(tr.q_nets["cop"].parameters())
    assert (before - after).abs().max() > 1e-7


def test_trainer_iql_changes_each_q_independently() -> None:
    tr = _make_trainer("iql")
    cop_before = next(tr.q_nets["cop"].parameters()).clone()
    thief_before = next(tr.q_nets["thief"].parameters()).clone()
    tr.train(n_episodes=4)
    assert (cop_before - next(tr.q_nets["cop"].parameters())).abs().max() > 1e-7
    assert (thief_before - next(tr.q_nets["thief"].parameters())).abs().max() > 1e-7


def test_trainer_epsilon_decays_with_episode_count() -> None:
    tr = _make_trainer("qmix")
    eps_at_start = tr.eps_schedule.at(0)
    eps_at_50 = tr.eps_schedule.at(50)
    eps_at_100 = tr.eps_schedule.at(100)
    assert eps_at_start > eps_at_50 > eps_at_100 or eps_at_100 == tr.eps_schedule.final


def test_trainer_skips_learn_before_warmup() -> None:
    tr = _make_trainer("qmix")
    info = tr.learn_step()
    assert info["skipped"] is True


def test_trainer_global_state_dim_matches_buffer() -> None:
    tr = _make_trainer("qmix")
    assert tr.buffer.state_dim == tr.env.global_state().shape[0]


def test_trainer_uses_cpu_device() -> None:
    tr = _make_trainer("qmix")
    assert tr.device == torch.device("cpu")
