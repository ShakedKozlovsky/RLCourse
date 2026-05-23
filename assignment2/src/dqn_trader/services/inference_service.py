"""InferenceService — single-state decision API for GUI / CLI / notebooks.

Takes the *latest* market window (already feature-engineered + scaled) plus
the current portfolio state, builds the 10-channel observation, runs the
agent's online network in inference mode, and returns the chosen action,
the full Q-value vector, and a softmax-confidence reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dqn_trader.services.dqn_agent import DQNAgent
from dqn_trader.shared.types import Action


@dataclass(frozen=True)
class Decision:
    """A single inference call's payload — readable both in code and in the GUI."""

    action: Action
    q_values: np.ndarray  # shape (n_actions,)
    confidence: float  # softmax(Q)[action_index] ∈ (0, 1)


class InferenceService:
    """Stateless wrapper. Constructed with an already-loaded DQNAgent."""

    def __init__(self, agent: DQNAgent):
        self._agent = agent

    def decide(self, market_window: np.ndarray, position: int, pnl_unrealised_scaled: float) -> Decision:
        """Produce a decision for the *next* trading day given the latest window."""
        if market_window.ndim != 2 or market_window.shape[1] != 8:
            raise ValueError(f"expected (window, 8) market window, got {market_window.shape}")
        obs = self._assemble(market_window, position, pnl_unrealised_scaled)
        x = torch.from_numpy(obs).unsqueeze(0).to(self._agent.device, dtype=torch.float32)
        with torch.no_grad():
            q = self._agent.online(x).squeeze(0)
            probs = torch.softmax(q, dim=0)
        action_idx = int(torch.argmax(q).item())
        return Decision(
            action=Action(action_idx),
            q_values=q.cpu().numpy(),
            confidence=float(probs[action_idx].item()),
        )

    @staticmethod
    def _assemble(market: np.ndarray, position: int, pnl_scaled: float) -> np.ndarray:
        window, _ = market.shape
        pos_col = np.full((window, 1), float(position), dtype=np.float32)
        pnl_col = np.full((window, 1), float(pnl_scaled), dtype=np.float32)
        return np.concatenate([market.astype(np.float32), pos_col, pnl_col], axis=1)
