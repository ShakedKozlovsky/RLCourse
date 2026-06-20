"""PyTorch parameter init helpers — orthogonal init for hidden layers, small-gain
final layer for the actor head (Engstrom 2020 / DDPG convention)."""

from __future__ import annotations

import math

import torch.nn as nn


def init_hidden(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_actor_head(layer: nn.Linear, gain: float = 0.1) -> None:
    """Small-but-not-tiny gain on the actor head.

    The original DDPG paper used near-zero gain (0.003) to avoid tanh saturation.
    In our cleaning-robot setting the initial policy needs to MOVE — too-small a
    gain makes the agent essentially stationary, the buffer fills with no-motion
    transitions, and the critic learns Q(s, 0) only. We default to 0.1 (1σ
    output magnitude ≈ 0.1 after the body's ReLU+orthogonal stack), which keeps
    the network well away from tanh saturation while still producing meaningful
    forward velocity from step 0."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_critic_head(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
