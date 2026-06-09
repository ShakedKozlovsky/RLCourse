"""Orthogonal initialisation — the PPO recipe (Engstrom et al. 2020).

Hidden tanh layers: gain = sqrt(2). Actor output: gain = 0.01 (small initial
actions). Critic output: gain = 1.0 (values can have any scale). Biases zeroed.
"""

from __future__ import annotations

import math

from torch import nn

HIDDEN_GAIN = math.sqrt(2.0)
ACTOR_HEAD_GAIN = 0.01
CRITIC_HEAD_GAIN = 1.0


def orthogonal_(layer: nn.Linear, gain: float) -> None:
    """In-place orthogonal init with zero bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_mlp(layers: list[nn.Linear], head_gain: float) -> None:
    """Initialise a list of ``nn.Linear`` — all but the last with hidden gain."""
    for layer in layers[:-1]:
        orthogonal_(layer, HIDDEN_GAIN)
    orthogonal_(layers[-1], head_gain)
