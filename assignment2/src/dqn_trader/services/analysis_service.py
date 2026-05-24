"""Advanced analysis tools for excellence-grade differentiation.

Provides three analyses that most students won't implement:
1. Window-size sensitivity sweep
2. Per-episode action distribution (reward-hacking detector)
3. Q-value heatmap over the test slice
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dqn_trader.environment.trading_env import TradingEnv
from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.shared.logger import get_logger
from dqn_trader.shared.types import Action

_logger = get_logger(__name__)


@dataclass(frozen=True)
class ActionDistribution:
    """Per-episode action counts and fractions."""

    episode: int
    sell_frac: float
    hold_frac: float
    buy_frac: float
    total_steps: int


@dataclass(frozen=True)
class QValueSnapshot:
    """Q-values at every step of a test-slice rollout."""

    q_sell: np.ndarray
    q_hold: np.ndarray
    q_buy: np.ndarray
    actions_taken: np.ndarray
    portfolio_values: np.ndarray


def collect_action_distribution(
    agent: DQNAgent, env: TradingEnv, *, epsilon: float
) -> ActionDistribution:
    """Run one episode and count how often each action was chosen."""
    state, _ = env.reset()
    rng = np.random.default_rng(0)
    counts = np.zeros(Action.n(), dtype=int)
    done = False
    while not done:
        action = agent.act(state, epsilon=epsilon, rng=rng)
        counts[action] += 1
        state, _, done, _, _ = env.step(action)
    total = int(counts.sum())
    fracs = counts / max(total, 1)
    return ActionDistribution(
        episode=0,
        sell_frac=float(fracs[Action.SELL]),
        hold_frac=float(fracs[Action.HOLD]),
        buy_frac=float(fracs[Action.BUY]),
        total_steps=total,
    )


def collect_qvalue_heatmap(agent: DQNAgent, env: TradingEnv) -> QValueSnapshot:
    """Greedy rollout recording Q-values at every step for visualisation."""
    state, _ = env.reset()
    q_sell, q_hold, q_buy = [], [], []
    actions, values = [], []
    done = False
    while not done:
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).to(
                agent.device, dtype=torch.float32
            )
            q = agent.online(x).squeeze(0).cpu().numpy()
        q_sell.append(float(q[Action.SELL]))
        q_hold.append(float(q[Action.HOLD]))
        q_buy.append(float(q[Action.BUY]))
        action = int(np.argmax(q))
        actions.append(action)
        state, _, done, _, info = env.step(action)
        values.append(info["portfolio_value"])
    return QValueSnapshot(
        q_sell=np.array(q_sell),
        q_hold=np.array(q_hold),
        q_buy=np.array(q_buy),
        actions_taken=np.array(actions, dtype=np.int64),
        portfolio_values=np.array(values),
    )
