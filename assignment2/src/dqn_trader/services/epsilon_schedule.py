"""Linear anneal schedules — ε (decreasing) and PER β (increasing)."""

from __future__ import annotations


class LinearSchedule:
    """Linearly interpolate from ``start`` to ``end`` over ``decay_steps``.

    The direction is inferred from the values; both increasing (β: 0.4 → 1.0)
    and decreasing (ε: 1.0 → 0.05) usages are supported. Outside the decay
    window the value is clamped to start (before) and end (after).
    """

    def __init__(self, start: float, end: float, decay_steps: int):
        if decay_steps < 1:
            raise ValueError("decay_steps must be >= 1")
        self._start = float(start)
        self._end = float(end)
        self._decay = int(decay_steps)

    def value(self, step: int) -> float:
        if step <= 0:
            return self._start
        if step >= self._decay:
            return self._end
        frac = step / self._decay
        return self._start + frac * (self._end - self._start)


# Two aliases keep call sites self-documenting without code duplication.
EpsilonSchedule = LinearSchedule
BetaSchedule = LinearSchedule
