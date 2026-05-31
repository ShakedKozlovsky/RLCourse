"""Audit finding #1: validate the LSTM against trivial baselines.

A trained world model is only useful if it beats simple baselines. This
service compares the LSTM's prediction error against:

- **Persistence**: ``s_{t+1} = s_t`` (do nothing)
- **Linear regression**: a closed-form lstsq fit ``s_{t+1} = A·[s_t; a_t] + b``
  trained on the same windows the LSTM saw

…and reports the multi-step rollout MSE at horizons {1, 7, 28} so we can
see how the world-model error compounds when used inside the policy loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nn_f

from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.shared.types import Action


@dataclass(frozen=True)
class WorldModelReport:
    """Per-baseline + per-horizon MSE summary."""

    persistence_one_step_mse: float
    linear_one_step_mse: float
    lstm_one_step_mse: float
    lstm_rollout_mse: dict[int, float]  # horizon -> mse
    n_test: int

    def to_dict(self) -> dict:
        return {
            "persistence_one_step_mse": self.persistence_one_step_mse,
            "linear_one_step_mse": self.linear_one_step_mse,
            "lstm_one_step_mse": self.lstm_one_step_mse,
            "lstm_rollout_mse": {str(k): v for k, v in self.lstm_rollout_mse.items()},
            "n_test": self.n_test,
        }


class WorldModelEvaluator:
    """Compute baseline MSEs + LSTM rollout error on a held-out trajectory slice."""

    def __init__(self, window_size: int = 7, test_pct: float = 0.2):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < test_pct < 1.0:
            raise ValueError("test_pct must be in (0, 1)")
        self._W = int(window_size)
        self._test_pct = float(test_pct)

    def evaluate(
        self, model: LSTMWorldModel, states: np.ndarray, actions: np.ndarray,
        rollout_horizons: tuple[int, ...] = (1, 7, 28),
    ) -> WorldModelReport:
        t = states.shape[0]
        n_train = int((1.0 - self._test_pct) * t)
        # Persistence: predict next state == current state.
        persistence_mse = float(((states[1:] - states[:-1]) ** 2).mean())
        # Linear baseline trained on the train slice.
        linear_mse = self._linear_baseline_mse(
            states[:n_train], actions[:n_train],
            states[n_train:], actions[n_train:],
        )
        # LSTM one-step + multi-step rollout error on the test slice.
        lstm_one = self._lstm_one_step_mse(model, states[n_train:], actions[n_train:])
        rollout = {h: self._lstm_rollout_mse(model, states[n_train:], actions[n_train:], h)
                   for h in rollout_horizons}
        return WorldModelReport(
            persistence_one_step_mse=persistence_mse,
            linear_one_step_mse=linear_mse,
            lstm_one_step_mse=lstm_one,
            lstm_rollout_mse=rollout,
            n_test=states.shape[0] - n_train,
        )

    def _linear_baseline_mse(
        self, s_train: np.ndarray, a_train: np.ndarray,
        s_test: np.ndarray, a_test: np.ndarray,
    ) -> float:
        """Closed-form OLS on ``[s_t; one_hot(a_t)] → s_{t+1}``."""
        d_s, d_a = s_train.shape[1], Action.n()
        x_train = np.zeros((s_train.shape[0] - 1, d_s + d_a), dtype=np.float64)
        x_train[:, :d_s] = s_train[:-1]
        x_train[np.arange(s_train.shape[0] - 1), d_s + a_train[:-1]] = 1.0
        y_train = s_train[1:].astype(np.float64)
        # OLS with bias absorbed: append 1s column
        x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        # Predict on test
        x_test = np.zeros((s_test.shape[0] - 1, d_s + d_a), dtype=np.float64)
        x_test[:, :d_s] = s_test[:-1]
        x_test[np.arange(s_test.shape[0] - 1), d_s + a_test[:-1]] = 1.0
        x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
        y_pred = x_test @ coef
        y_test = s_test[1:].astype(np.float64)
        return float(((y_pred - y_test) ** 2).mean())

    def _lstm_one_step_mse(
        self, model: LSTMWorldModel, states: np.ndarray, actions: np.ndarray
    ) -> float:
        model.eval()
        n_windows = states.shape[0] - self._W
        if n_windows < 1:
            return float("nan")
        s = torch.from_numpy(states).float()
        a = torch.from_numpy(actions).long()
        preds = []
        targets = []
        with torch.no_grad():
            for i in range(n_windows):
                x = model.encode_inputs(
                    s[i : i + self._W].unsqueeze(0), a[i : i + self._W].unsqueeze(0)
                )
                preds.append(model(x).squeeze(0))
                targets.append(s[i + self._W])
        return float(nn_f.mse_loss(torch.stack(preds), torch.stack(targets)).item())

    def _lstm_rollout_mse(
        self, model: LSTMWorldModel, states: np.ndarray, actions: np.ndarray,
        horizon: int,
    ) -> float:
        """Roll the LSTM forward ``horizon`` steps using its own outputs."""
        model.eval()
        if states.shape[0] < self._W + horizon:
            return float("nan")
        s_window = states[: self._W].astype(np.float32).copy()
        squared = 0.0
        n = 0
        with torch.no_grad():
            for h in range(horizon):
                a_window = actions[h : h + self._W]
                x = model.encode_inputs(
                    torch.from_numpy(s_window).float().unsqueeze(0),
                    torch.from_numpy(a_window).long().unsqueeze(0),
                )
                pred = model(x).squeeze(0).cpu().numpy().astype(np.float32)
                target = states[self._W + h].astype(np.float32)
                squared += float(((pred - target) ** 2).mean())
                n += 1
                s_window = np.concatenate([s_window[1:], pred[None, :]], axis=0)
        return squared / max(1, n)
