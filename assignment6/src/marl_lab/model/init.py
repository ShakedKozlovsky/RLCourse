"""PyTorch parameter init helpers — orthogonal init (Engstrom 2020 / standard
DDPG-PPO recipe). Carried-over from A4/A5."""

from __future__ import annotations

import math

import torch.nn as nn


def init_hidden(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    """Orthogonal init for hidden layers — gain √2 (Engstrom 2020 / community default)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_q_head(layer: nn.Linear, gain: float = 1.0) -> None:
    """Standard-gain orthogonal init for the Q-value head."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
