"""Layer 8 + 9 — centralised buffer + ε-greedy + schedule tests."""

from __future__ import annotations

import numpy as np
import pytest

from marl_lab.memory.centralised_buffer import CentralisedReplayBuffer
from marl_lab.noise.epsilon_greedy import select_action
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.shared.types import EpisodeSequence, Transition

# ----- Centralised buffer -----

def _dummy_episode(length: int, state_dim: int = 4, obs_dim: int = 6) -> EpisodeSequence:
    ep = EpisodeSequence()
    for t in range(length):
        ep.transitions.append(Transition(
            global_state=np.full(state_dim, float(t), dtype=np.float32),
            joint_obs={"cop": np.full(obs_dim, float(t), dtype=np.float32),
                       "thief": np.full(obs_dim, float(t) + 0.5, dtype=np.float32)},
            joint_action={"cop": t % 5, "thief": t % 4},
            joint_reward={"cop": -0.05, "thief": -0.01},
            next_global_state=np.full(state_dim, float(t) + 1.0, dtype=np.float32),
            next_joint_obs={"cop": np.full(obs_dim, float(t) + 1.0, dtype=np.float32),
                            "thief": np.full(obs_dim, float(t) + 1.5, dtype=np.float32)},
            done=(t == length - 1),
        ))
    return ep


def test_buffer_validates_capacity() -> None:
    with pytest.raises(ValueError):
        CentralisedReplayBuffer(capacity=0, max_seq_len=25, state_dim=4, obs_dim=6,
                                 n_actions_per_agent={"cop": 6, "thief": 5})


def test_buffer_validates_max_seq_len() -> None:
    with pytest.raises(ValueError):
        CentralisedReplayBuffer(capacity=4, max_seq_len=0, state_dim=4, obs_dim=6,
                                 n_actions_per_agent={"cop": 6, "thief": 5})


def test_buffer_empty_starts_at_size_zero() -> None:
    buf = CentralisedReplayBuffer(capacity=4, max_seq_len=25, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5})
    assert len(buf) == 0


def test_buffer_push_grows_up_to_capacity() -> None:
    buf = CentralisedReplayBuffer(capacity=3, max_seq_len=25, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5})
    for _ in range(5):
        buf.push(_dummy_episode(10))
    assert len(buf) == 3       # capped


def test_buffer_sample_shape_contract() -> None:
    buf = CentralisedReplayBuffer(capacity=8, max_seq_len=25, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    for _ in range(8):
        buf.push(_dummy_episode(10))
    b = buf.sample(batch_size=4)
    assert b.state_seq.shape == (4, 25, 4)
    assert b.next_state_seq.shape == (4, 25, 4)
    assert b.obs_seq["cop"].shape == (4, 25, 6)
    assert b.action_seq["thief"].shape == (4, 25)
    assert b.reward_seq["cop"].shape == (4, 25)
    assert b.done_seq.shape == (4, 25)
    assert b.mask.shape == (4, 25)


def test_buffer_padding_mask_correct() -> None:
    buf = CentralisedReplayBuffer(capacity=4, max_seq_len=25, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    buf.push(_dummy_episode(7))      # only 7 valid timesteps
    b = buf.sample(batch_size=1)
    assert b.mask[0, :7].sum() == 7
    assert b.mask[0, 7:].sum() == 0   # rest is padding


def test_buffer_sample_too_large_raises() -> None:
    buf = CentralisedReplayBuffer(capacity=4, max_seq_len=25, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5})
    buf.push(_dummy_episode(5))
    with pytest.raises(ValueError):
        buf.sample(batch_size=5)


def test_buffer_truncates_at_max_seq_len() -> None:
    buf = CentralisedReplayBuffer(capacity=4, max_seq_len=10, state_dim=4,
                                    obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5},
                                    rng=np.random.default_rng(0))
    buf.push(_dummy_episode(30))    # longer than max_seq_len
    b = buf.sample(batch_size=1)
    assert b.mask[0].sum() == 10


def test_buffer_same_rng_same_indices() -> None:
    bufs = [CentralisedReplayBuffer(capacity=8, max_seq_len=10, state_dim=4,
                                      obs_dim=6, n_actions_per_agent={"cop": 6, "thief": 5},
                                      rng=np.random.default_rng(123))
            for _ in range(2)]
    for buf in bufs:
        for _ in range(8):
            buf.push(_dummy_episode(5))
    a = bufs[0].sample(4).state_seq
    b = bufs[1].sample(4).state_seq
    np.testing.assert_array_equal(a, b)


# ----- ε-greedy -----

def test_select_action_epsilon_zero_argmax() -> None:
    q = np.array([0.1, 0.5, 0.3, 0.2])
    a = select_action(q, epsilon=0.0, rng=np.random.default_rng(0))
    assert a == 1   # argmax


def test_select_action_epsilon_one_random() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    samples = [select_action(q, epsilon=1.0, rng=rng) for _ in range(200)]
    # Distribution should be roughly uniform → not just 0
    assert len(set(samples)) > 1


def test_select_action_respects_mask() -> None:
    q = np.array([100.0, 0.0, 0.0])
    mask = np.array([False, True, True])     # action 0 illegal even though best
    a = select_action(q, epsilon=0.0, rng=np.random.default_rng(0), action_mask=mask)
    assert a != 0


def test_select_action_no_legal_actions_raises() -> None:
    q = np.zeros(3)
    mask = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError):
        select_action(q, epsilon=0.5, rng=np.random.default_rng(0), action_mask=mask)


# ----- ε schedule -----

def test_schedule_initial_and_final() -> None:
    s = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=1000)
    assert s.at(0) == pytest.approx(1.0)
    assert s.at(1000) == pytest.approx(0.05)
    assert s.at(5000) == pytest.approx(0.05)


def test_schedule_midpoint() -> None:
    s = LinearEpsilonSchedule(initial=1.0, final=0.0, decay_steps=1000)
    assert s.at(500) == pytest.approx(0.5)


def test_schedule_validates_decay_steps() -> None:
    with pytest.raises(ValueError):
        LinearEpsilonSchedule(initial=1.0, final=0.0, decay_steps=0)


def test_schedule_validates_epsilon_range() -> None:
    with pytest.raises(ValueError):
        LinearEpsilonSchedule(initial=1.5, final=0.0, decay_steps=100)
    with pytest.raises(ValueError):
        LinearEpsilonSchedule(initial=0.5, final=-0.1, decay_steps=100)
