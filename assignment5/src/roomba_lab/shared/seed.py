"""Single entry point for deterministic seeding across Python + NumPy + PyTorch.

Calling :func:`set_global_seed` before any agent / env construction is sufficient
to make the full training run reproducible bit-for-bit on CPU (assuming
single-threaded data ops, which we enforce in tests)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU and CUDA — uniformly."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
