"""Linear ε schedule — anneals exploration probability from `initial` to
`final` over `decay_steps` global steps, then clamps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearEpsilonSchedule:
    """Linear interpolation: at(0) = initial, at(decay_steps) = final, clamped."""
    initial: float
    final: float
    decay_steps: int

    def __post_init__(self) -> None:
        if self.decay_steps <= 0:
            raise ValueError(f"decay_steps must be > 0, got {self.decay_steps}")
        if not (0.0 <= self.final <= 1.0 and 0.0 <= self.initial <= 1.0):
            raise ValueError("epsilon values must lie in [0, 1]")

    def at(self, step: int) -> float:
        """Linearly interpolate ε between (initial, final) over decay_steps."""
        frac = min(1.0, max(0.0, step / self.decay_steps))
        return float(self.initial + (self.final - self.initial) * frac)
