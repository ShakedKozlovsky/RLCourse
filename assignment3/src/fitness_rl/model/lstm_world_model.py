"""LSTM world model: learns P(s_{t+1} | s_t, a_t, h_t) over rolling windows.

This *is* the environment dynamics for REINFORCE + A2C — Ha & Schmidhuber's
"World Models" idea (2018): train a recurrent transition model, then train a
policy inside it. See ``docs/PRD_lstm_world_model.md``.

Input to forward():  (B, W, D_s + D_a)  — state+one-hot-action per timestep
Output of forward(): (B, D_s)           — predicted next state after the window
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_f

from fitness_rl.data.feature_engineer import STATE_DIM
from fitness_rl.shared.types import Action

TransitionFn = Callable[[np.ndarray, int], np.ndarray]


class LSTMWorldModel(nn.Module):
    """1-layer LSTM(D_s+D_a → hidden) + Linear(hidden → D_s) head."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be >= 1")
        self.state_dim = STATE_DIM
        self.action_dim = Action.n()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.state_dim + self.action_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(self.hidden_size, self.state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, W, D_s+D_a); returns (B, D_s)."""
        if x.dim() != 3 or x.size(-1) != self.state_dim + self.action_dim:
            raise ValueError(
                f"expected (B, W, {self.state_dim + self.action_dim}), got {tuple(x.shape)}"
            )
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

    @staticmethod
    def encode_inputs(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Concatenate states (..., D_s) and one-hot-encoded actions (...) → (..., D_s+D_a)."""
        if states.dim() < 2 or states.size(-1) != STATE_DIM:
            raise ValueError(f"states must have last dim {STATE_DIM}; got {tuple(states.shape)}")
        if actions.shape != states.shape[:-1]:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} != states[..., :-1] {states.shape[:-1]}"
            )
        one_hot = nn_f.one_hot(actions.long(), num_classes=Action.n()).to(states.dtype)
        return torch.cat([states, one_hot], dim=-1)

    def save(self, path: Path) -> None:
        """Persist weights + architecture hyperparameters for round-trip reload."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LSTMWorldModel:
        """Restore a saved model from ``path``."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(hidden_size=int(blob["hidden_size"]), num_layers=int(blob["num_layers"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model

    def as_transition_fn(self, window_size: int, warmup_state: np.ndarray) -> TransitionFn:
        """Return a stateful ``(state, action) -> next_state`` closure.

        Maintains a rolling window of the last ``window_size`` (state, action) pairs and
        runs one LSTM forward pass per call. ``warmup_state`` seeds the window so the
        first call already sees a full context — duplicate the env's initial state.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        history: deque[tuple[np.ndarray, int]] = deque(
            [(warmup_state.astype(np.float32, copy=True), int(Action.REST))] * window_size,
            maxlen=window_size,
        )

        def step(state: np.ndarray, action: int) -> np.ndarray:
            history.append((state.astype(np.float32, copy=True), int(action)))
            states = torch.from_numpy(np.stack([s for s, _ in history])).float().unsqueeze(0)
            actions = torch.tensor([[a for _, a in history]], dtype=torch.long)
            with torch.no_grad():
                x = self.encode_inputs(states, actions)
                return self(x).squeeze(0).cpu().numpy().astype(np.float32)

        return step
