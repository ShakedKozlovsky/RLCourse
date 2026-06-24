"""Deterministic seeding across Python + NumPy + PyTorch.

Same shape as A4/A5. Calling :func:`set_global_seed` before any agent / env
construction is sufficient to make the full training run reproducible
bit-for-bit on CPU."""

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
