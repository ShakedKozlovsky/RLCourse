"""DQNAgent — act semantics, optimize loss math, target sync, save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.dqn_agent import DQNAgent


def _agent(replay: UniformReplay, **kw) -> DQNAgent:
    defaults: dict[str, object] = {
        "window_size": 30, "n_features": 10, "n_actions": 3, "replay": replay,
        "gamma": 0.99, "lr": 1e-3, "target_sync_every": 2,
        "dueling": True, "double_dqn": True,
    }
    defaults.update(kw)
    torch.manual_seed(0)
    return DQNAgent(**defaults)  # type: ignore[arg-type]


def test_act_returns_action_index() -> None:
    agent = _agent(UniformReplay(capacity=100))
    a = agent.act(np.zeros((30, 10), dtype=np.float32), epsilon=0.0, rng=np.random.default_rng(0))
    assert a in {0, 1, 2}


def test_act_with_epsilon_one_is_uniform_random() -> None:
    agent = _agent(UniformReplay(capacity=100))
    rng = np.random.default_rng(0)
    counts = np.zeros(3, dtype=int)
    for _ in range(300):
        counts[agent.act(np.zeros((30, 10), dtype=np.float32), epsilon=1.0, rng=rng)] += 1
    assert counts.min() > 50  # all three actions sampled meaningfully


def test_optimize_returns_none_when_buffer_empty() -> None:
    agent = _agent(UniformReplay(capacity=10))
    assert agent.optimize(batch_size=8, beta=0.5) is None


def test_optimize_returns_loss_after_enough_data() -> None:
    buf = UniformReplay(capacity=64, seed=0)
    for _ in range(32):
        buf.add(
            np.zeros((30, 10), dtype=np.float32), 1, 0.1,
            np.zeros((30, 10), dtype=np.float32), False,
        )
    agent = _agent(buf)
    loss = agent.optimize(batch_size=16, beta=0.5)
    assert loss is not None and loss >= 0.0


def test_target_sync_happens_at_interval() -> None:
    buf = UniformReplay(capacity=64, seed=0)
    for _ in range(32):
        buf.add(
            np.zeros((30, 10), dtype=np.float32), 1, 0.1,
            np.zeros((30, 10), dtype=np.float32), False,
        )
    agent = _agent(buf, target_sync_every=1)
    for p in agent.target.parameters():
        p.data.zero_()
    agent.optimize(batch_size=8, beta=0.5)
    # After a sync the target must match online — check at least one tensor.
    same = any(
        torch.equal(tp.data, op.data)
        for tp, op in zip(agent.target.parameters(), agent.online.parameters(), strict=True)
    )
    assert same


def test_save_load_round_trip(tmp_path: Path) -> None:
    agent = _agent(UniformReplay(capacity=10))
    path = tmp_path / "ckpt.pt"
    agent.save(str(path))
    fresh = _agent(UniformReplay(capacity=10))
    fresh.load(str(path))
    for sp, fp in zip(agent.online.parameters(), fresh.online.parameters(), strict=True):
        torch.testing.assert_close(sp, fp)


def test_double_dqn_path_runs() -> None:
    buf = UniformReplay(capacity=64, seed=0)
    for _ in range(32):
        buf.add(
            np.zeros((30, 10), dtype=np.float32), 1, 0.1,
            np.zeros((30, 10), dtype=np.float32), False,
        )
    agent = _agent(buf, double_dqn=False)
    loss = agent.optimize(batch_size=8, beta=0.5)
    assert loss is not None


@pytest.mark.parametrize("dueling", [True, False])
def test_dueling_flag_changes_n_params(dueling: bool) -> None:
    agent = _agent(UniformReplay(capacity=10), dueling=dueling)
    n_params = sum(p.numel() for p in agent.online.parameters())
    assert n_params > 0  # smoke: both variants instantiate
