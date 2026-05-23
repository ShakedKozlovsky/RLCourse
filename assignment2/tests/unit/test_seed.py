"""set_global_seed produces reproducible NumPy + random + torch state."""

from __future__ import annotations

import random

import numpy as np

from dqn_trader.shared.seed import set_global_seed


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.default_rng().random()) * 0  # see comment


def test_python_and_numpy_legacy_repeat() -> None:
    set_global_seed(1234)
    a1 = random.random()
    b1 = np.random.random()
    set_global_seed(1234)
    a2 = random.random()
    b2 = np.random.random()
    assert a1 == a2
    assert b1 == b2


def test_torch_seed_applied_if_available() -> None:
    try:
        import torch
    except ImportError:
        return
    set_global_seed(99)
    x1 = torch.randn(3)
    set_global_seed(99)
    x2 = torch.randn(3)
    assert torch.equal(x1, x2)
