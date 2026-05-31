"""LSTMWorldModel — forward shape, finiteness, save/load, transition-fn adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fitness_rl.model.lstm_world_model import LSTMWorldModel
from fitness_rl.shared.types import Action


def test_invalid_init_raises() -> None:
    with pytest.raises(ValueError):
        LSTMWorldModel(hidden_size=0)
    with pytest.raises(ValueError):
        LSTMWorldModel(num_layers=0)


def test_forward_shape() -> None:
    model = LSTMWorldModel(hidden_size=8)
    x = torch.randn(4, 7, model.state_dim + model.action_dim)
    out = model(x)
    assert out.shape == (4, model.state_dim)


def test_forward_finite_outputs() -> None:
    torch.manual_seed(0)
    model = LSTMWorldModel(hidden_size=8)
    x = torch.randn(2, 7, model.state_dim + model.action_dim)
    out = model(x)
    assert torch.isfinite(out).all()


def test_forward_rejects_bad_shape() -> None:
    model = LSTMWorldModel(hidden_size=8)
    with pytest.raises(ValueError):
        model(torch.randn(4, 7, 10))  # wrong feature dim
    with pytest.raises(ValueError):
        model(torch.randn(7, 21))  # missing batch dim


def test_encode_inputs_concatenates_correctly() -> None:
    states = torch.randn(3, 4, LSTMWorldModel().state_dim)
    actions = torch.tensor([[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1]])
    x = LSTMWorldModel.encode_inputs(states, actions)
    assert x.shape == (3, 4, LSTMWorldModel().state_dim + Action.n())
    # state portion preserved
    assert torch.allclose(x[..., : LSTMWorldModel().state_dim], states)
    # action one-hot sums to 1 per timestep
    assert torch.allclose(x[..., LSTMWorldModel().state_dim :].sum(dim=-1),
                          torch.ones(3, 4))


def test_encode_inputs_validates_shapes() -> None:
    with pytest.raises(ValueError):
        LSTMWorldModel.encode_inputs(torch.randn(3, 10), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError):
        LSTMWorldModel.encode_inputs(
            torch.randn(3, 4, 16), torch.tensor([[0, 1], [2, 3], [4, 0]])
        )  # actions shape mismatch


def test_save_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original = LSTMWorldModel(hidden_size=16, num_layers=1)
    original.eval()
    x = torch.randn(2, 7, original.state_dim + original.action_dim)
    with torch.no_grad():
        before = original(x)
    ckpt = tmp_path / "wm.pt"
    original.save(ckpt)
    restored = LSTMWorldModel.load(ckpt)
    assert restored.hidden_size == 16
    with torch.no_grad():
        after = restored(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_as_transition_fn_returns_callable() -> None:
    model = LSTMWorldModel(hidden_size=8)
    model.eval()
    warmup = np.zeros(model.state_dim, dtype=np.float32)
    tfn = model.as_transition_fn(window_size=7, warmup_state=warmup)
    state = np.zeros(model.state_dim, dtype=np.float32)
    next_state = tfn(state, int(Action.PUSH))
    assert next_state.shape == (model.state_dim,)
    assert next_state.dtype == np.float32
    assert np.all(np.isfinite(next_state))


def test_as_transition_fn_invalid_window_raises() -> None:
    model = LSTMWorldModel(hidden_size=8)
    with pytest.raises(ValueError):
        model.as_transition_fn(window_size=0, warmup_state=np.zeros(16, dtype=np.float32))
