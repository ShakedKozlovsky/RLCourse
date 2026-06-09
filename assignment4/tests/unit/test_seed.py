"""set_global_seed — Python + NumPy + PyTorch determinism."""

from __future__ import annotations

import random

import numpy as np
import torch

from proximal_lab.shared.seed import set_global_seed


def test_python_random_seeded() -> None:
    set_global_seed(7)
    a = random.random()
    set_global_seed(7)
    b = random.random()
    assert a == b


def test_numpy_seeded() -> None:
    set_global_seed(11)
    a = float(np.random.random())
    set_global_seed(11)
    b = float(np.random.random())
    assert a == b


def test_torch_seeded() -> None:
    set_global_seed(23)
    a = torch.rand(5)
    set_global_seed(23)
    b = torch.rand(5)
    assert torch.allclose(a, b)
