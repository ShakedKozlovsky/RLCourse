"""Z-score scaler that fits on train and applies to val/test.

Why our own and not sklearn: zero deps, JSON-serialisable state, makes the
no-leakage contract explicit in the test suite.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ScalerState:
    """Persisted fit result. Saved next to model checkpoints."""

    columns: list[str]
    mean: list[float]
    std: list[float]


class ZScoreScaler:
    """Per-column z-scoring. ``fit`` is called exactly once, on train only."""

    def __init__(self) -> None:
        self._state: ScalerState | None = None

    def fit(self, df: pd.DataFrame) -> ZScoreScaler:
        if self._state is not None:
            raise RuntimeError("Scaler already fitted; create a new one per run")
        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=0).replace(0.0, 1.0)
        self._state = ScalerState(
            columns=list(df.columns),
            mean=mean.tolist(),
            std=std.tolist(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._state is None:
            raise RuntimeError("Scaler not fitted")
        if list(df.columns) != self._state.columns:
            raise ValueError(
                f"Column mismatch: scaler={self._state.columns}, df={list(df.columns)}"
            )
        mean = np.asarray(self._state.mean)
        std = np.asarray(self._state.std)
        arr = (df.to_numpy() - mean) / std
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    @property
    def state(self) -> ScalerState:
        if self._state is None:
            raise RuntimeError("Scaler not fitted")
        return self._state

    def save(self, path: Path) -> None:
        if self._state is None:
            raise RuntimeError("Scaler not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self._state), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ZScoreScaler:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj._state = ScalerState(**data)
        return obj
