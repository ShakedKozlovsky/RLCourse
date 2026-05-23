"""Global seed plumbing — one place to make a run deterministic."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA if available).

    Why a thin wrapper: every layer that introduces randomness (env init,
    replay sampling, model init, dropout, exploration) must agree on the
    seed. Centralising it makes ``seed=X`` in config truly reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
