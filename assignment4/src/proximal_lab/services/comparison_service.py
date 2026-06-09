"""Cross-config comparison — aggregate seeds with mean ± 95 % CI."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CellSummary:
    """One config × seeds aggregate — used for sweep cells + cross-env cells."""

    name: str
    n_seeds: int
    final_reward_mean: float
    final_reward_ci_95: float
    overall_reward_mean: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    """A full sweep / comparison result for a configurable list of cells."""

    label: str
    cells: list[CellSummary]

    def to_dict(self) -> dict:
        return {"label": self.label, "cells": [c.to_dict() for c in self.cells]}


def aggregate_with_ci(per_seed_finals: list[float], z: float = 1.96) -> tuple[float, float]:
    """Mean ± 95 % normal-approx CI of a list of per-seed final rewards."""
    arr = np.asarray(per_seed_finals, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    se = arr.std(ddof=1) / np.sqrt(n)
    return mean, float(z * se)


class ComparisonService:
    """Build a :class:`ComparisonReport` from a dict ``{cell_name: [per-seed finals]}``."""

    def report(self, label: str, per_cell_finals: dict[str, list[float]]) -> ComparisonReport:
        cells = []
        for name, finals in per_cell_finals.items():
            mean, ci = aggregate_with_ci(finals)
            cells.append(CellSummary(
                name=name, n_seeds=len(finals),
                final_reward_mean=mean, final_reward_ci_95=ci,
                overall_reward_mean=float(np.mean(finals)) if finals else 0.0,
            ))
        return ComparisonReport(label=label, cells=cells)
