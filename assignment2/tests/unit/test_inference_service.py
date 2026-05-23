"""InferenceService — decision schema and shape guards."""

from __future__ import annotations

import numpy as np
import pytest

from dqn_trader.memory.uniform_replay import UniformReplay
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.services.inference_service import Decision, InferenceService
from dqn_trader.shared.types import Action


def _agent() -> DQNAgent:
    return DQNAgent(
        window_size=30, n_features=10, n_actions=3,
        replay=UniformReplay(capacity=10), gamma=0.99, lr=1e-3,
        dueling=True, double_dqn=True,
    )


def test_decide_returns_valid_decision() -> None:
    svc = InferenceService(_agent())
    market = np.zeros((30, 8), dtype=np.float32)
    d = svc.decide(market, position=0, pnl_unrealised_scaled=0.0)
    assert isinstance(d, Decision)
    assert d.action in (Action.SELL, Action.HOLD, Action.BUY)
    assert d.q_values.shape == (3,)
    assert 0.0 < d.confidence <= 1.0


def test_decide_action_matches_argmax_of_q() -> None:
    svc = InferenceService(_agent())
    d = svc.decide(np.zeros((30, 8), dtype=np.float32), position=1, pnl_unrealised_scaled=0.01)
    assert int(d.action) == int(np.argmax(d.q_values))


def test_decide_rejects_wrong_shape() -> None:
    svc = InferenceService(_agent())
    with pytest.raises(ValueError):
        svc.decide(np.zeros((30, 10), dtype=np.float32), position=0, pnl_unrealised_scaled=0.0)
    with pytest.raises(ValueError):
        svc.decide(np.zeros(30, dtype=np.float32), position=0, pnl_unrealised_scaled=0.0)
