"""WorldModelService — window construction, training, early stopping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.services.world_model_service import TrainResult, WorldModelService
from fitness_rl.shared.types import Action


def _synthetic_trajectory(t: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Linear-drift state, cycled actions — easy to learn."""
    rng = np.random.default_rng(seed)
    states = np.zeros((t, 16), dtype=np.float32)
    for i in range(t):
        states[i, 0] = (i % 7) / 7.0
        states[i, 1:6] = rng.dirichlet(np.ones(5))
        states[i, 6] = 0.5
        states[i, 7] = i / t
        states[i, 8 + (i % 7)] = 1.0
        states[i, 15] = 1.0 if (i % 7) == 6 else 0.0
    actions = np.array([i % Action.n() for i in range(t)], dtype=np.int64)
    return states, actions


def test_invalid_init_raises() -> None:
    with pytest.raises(ValueError):
        WorldModelService(window_size=0)
    with pytest.raises(ValueError):
        WorldModelService(train_pct=0.0)
    with pytest.raises(ValueError):
        WorldModelService(train_pct=1.0)


def test_build_windows_shapes() -> None:
    states, actions = _synthetic_trajectory(t=20)
    svc = WorldModelService(window_size=7)
    x, y = svc.build_windows(states, actions)
    expected_n = 20 - 7
    assert x.shape == (expected_n, 7, 16 + Action.n())
    assert y.shape == (expected_n, 16)


def test_build_windows_targets_are_next_states() -> None:
    states, actions = _synthetic_trajectory(t=15)
    svc = WorldModelService(window_size=5)
    _, y = svc.build_windows(states, actions)
    # window 0 targets state[5], window 1 targets state[6], etc.
    for i in range(y.size(0)):
        assert torch.allclose(y[i], torch.from_numpy(states[i + 5]))


def test_build_windows_rejects_short_trajectory() -> None:
    states, actions = _synthetic_trajectory(t=5)
    with pytest.raises(ValueError):
        WorldModelService(window_size=7).build_windows(states, actions)


def test_build_windows_validates_shapes() -> None:
    svc = WorldModelService(window_size=3)
    with pytest.raises(ValueError):
        svc.build_windows(np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.int64))
    with pytest.raises(ValueError):
        svc.build_windows(np.zeros((10, 16), dtype=np.float32),
                          np.zeros(5, dtype=np.int64))


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    states, actions = _synthetic_trajectory(t=60)
    model = LSTMWorldModel(hidden_size=16)
    svc = WorldModelService(window_size=7, epochs=20, batch_size=8,
                            early_stop_patience=20, lr=1e-2)
    result = svc.train(model, states, actions)
    assert isinstance(result, TrainResult)
    assert len(result.train_losses) >= 1
    assert result.train_losses[-1] < result.train_losses[0]


def test_early_stopping_triggers_on_flat_loss() -> None:
    torch.manual_seed(0)
    states, actions = _synthetic_trajectory(t=30)
    model = LSTMWorldModel(hidden_size=8)
    # Tiny patience + zero lr → no improvement → early stop on first eligible epoch.
    svc = WorldModelService(window_size=7, epochs=50, batch_size=4,
                            early_stop_patience=2, lr=0.0)
    result = svc.train(model, states, actions)
    assert result.stopped_early
    assert len(result.train_losses) < 50


def test_tiny_trajectory_falls_back_to_train_as_val() -> None:
    torch.manual_seed(0)
    # 9 steps, window 7 → 2 windows. train_pct=0.8 → n_train=1, val=1.
    states, actions = _synthetic_trajectory(t=9)
    model = LSTMWorldModel(hidden_size=8)
    svc = WorldModelService(window_size=7, epochs=3, batch_size=2,
                            early_stop_patience=10, lr=1e-3)
    result = svc.train(model, states, actions)
    assert len(result.train_losses) == 3
