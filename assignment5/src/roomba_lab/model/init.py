"""PyTorch parameter init helpers — orthogonal init for hidden layers, small-gain
final layer for the actor head (Engstrom 2020 / DDPG convention)."""

from __future__ import annotations

import math

import torch.nn as nn


def init_hidden(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_actor_head(layer: nn.Linear, gain: float = 0.01) -> None:
    """Small gain on the actor head keeps initial actions near zero, avoiding
    early saturation against the tanh."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_critic_head(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
