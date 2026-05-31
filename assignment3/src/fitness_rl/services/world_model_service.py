"""Supervised training service for the LSTM world model.

Builds rolling (state, action) → next-state windows from a trajectory,
splits 80/20 chronologically, trains with MSE + Adam + early stopping.
See ``docs/PRD_lstm_world_model.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.shared.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Per-epoch training log for plotting and reporting."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    stopped_early: bool = False


class WorldModelService:
    """Fit an :class:`LSTMWorldModel` on rolling windows of a trajectory."""

    def __init__(
        self,
        window_size: int = 7,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        early_stop_patience: int = 10,
        train_pct: float = 0.8,
    ):
        if window_size < 1 or batch_size < 1 or epochs < 1:
            raise ValueError("window_size, batch_size, epochs must be >= 1")
        if not 0.0 < train_pct < 1.0:
            raise ValueError("train_pct must be in (0, 1)")
        self._W = int(window_size)
        self._lr = float(lr)
        self._bs = int(batch_size)
        self._epochs = int(epochs)
        self._patience = int(early_stop_patience)
        self._train_pct = float(train_pct)

    def build_windows(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X, y) of shapes (N, W, D_s+D_a) and (N, D_s)."""
        if states.ndim != 2:
            raise ValueError(f"states must be (T, D_s); got {states.shape}")
        if actions.shape != (states.shape[0],):
            raise ValueError(f"actions shape {actions.shape} != ({states.shape[0]},)")
        n_windows = states.shape[0] - self._W
        if n_windows < 1:
            raise ValueError(f"trajectory too short: need > {self._W} steps")
        states_t = torch.from_numpy(states).float()
        actions_t = torch.from_numpy(actions).long()
        x_list, y_list = [], []
        for i in range(n_windows):
            window_states = states_t[i : i + self._W]
            window_actions = actions_t[i : i + self._W]
            x_list.append(LSTMWorldModel.encode_inputs(window_states, window_actions))
            y_list.append(states_t[i + self._W])
        return torch.stack(x_list), torch.stack(y_list)

    def train(
        self, model: LSTMWorldModel, states: np.ndarray, actions: np.ndarray
    ) -> TrainResult:
        """Train ``model`` in-place; return per-epoch losses + best-epoch info."""
        x, y = self.build_windows(states, actions)
        n_train = max(1, int(self._train_pct * x.size(0)))
        x_tr, y_tr = x[:n_train], y[:n_train]
        x_va, y_va = x[n_train:], y[n_train:]
        if x_va.size(0) == 0:
            x_va, y_va = x_tr, y_tr  # fall back: tiny trajectory, no val split
        opt = optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()
        train_losses, val_losses = [], []
        best_val, best_epoch, since_best = float("inf"), -1, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for epoch in range(self._epochs):
            tr_loss = self._train_one_epoch(model, opt, loss_fn, x_tr, y_tr)
            va_loss = self._eval_loss(model, loss_fn, x_va, y_va)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            if va_loss < best_val - 1e-6:
                best_val, best_epoch, since_best = va_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
            if since_best >= self._patience:
                _logger.info("early stop at epoch %d (best=%d)", epoch, best_epoch)
                model.load_state_dict(best_state)
                return TrainResult(train_losses, val_losses, best_val, best_epoch, True)
        model.load_state_dict(best_state)
        return TrainResult(train_losses, val_losses, best_val, best_epoch, False)

    def _train_one_epoch(
        self,
        model: LSTMWorldModel,
        opt: optim.Optimizer,
        loss_fn: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        model.train()
        perm = torch.randperm(x.size(0))
        running, n = 0.0, 0
        for start in range(0, x.size(0), self._bs):
            idx = perm[start : start + self._bs]
            opt.zero_grad()
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            running += float(loss.item()) * idx.size(0)
            n += idx.size(0)
        return running / max(1, n)

    @staticmethod
    def _eval_loss(
        model: LSTMWorldModel, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        model.eval()
        with torch.no_grad():
            return float(loss_fn(model(x), y).item())
