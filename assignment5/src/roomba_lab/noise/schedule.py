"""Linear σ schedule — anneals the exploration noise standard deviation from
`initial` to `final` over `decay_steps` global steps, then clamps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearSigmaSchedule:
    initial: float
    final: float
    decay_steps: int

    def __post_init__(self) -> None:
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be > 0")
        if self.initial < 0.0 or self.final < 0.0:
            raise ValueError("initial / final must be >= 0")

    def at(self, step: int) -> float:
        frac = min(1.0, max(0.0, step / self.decay_steps))
        return float(self.initial + (self.final - self.initial) * frac)
