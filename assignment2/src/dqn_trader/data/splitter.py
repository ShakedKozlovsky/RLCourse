"""Chronological train/val/test splitter — no shuffling, no leakage."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Three chronological slices of a DataFrame and their date boundaries."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class ChronologicalSplitter:
    """Splits a date-indexed DataFrame into train/val/test by row order."""

    def __init__(self, train_pct: float, val_pct: float):
        if not 0.0 < train_pct < 1.0 or not 0.0 < val_pct < 1.0:
            raise ValueError("train_pct and val_pct must be in (0, 1)")
        if train_pct + val_pct >= 1.0:
            raise ValueError("train_pct + val_pct must be < 1.0")
        self._train_pct = train_pct
        self._val_pct = val_pct

    def split(self, df: pd.DataFrame) -> Split:
        """Split a date-indexed DataFrame into chronological train/val/test."""
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame")
        n = len(df)
        n_train = int(round(n * self._train_pct))
        n_val = int(round(n * self._val_pct))
        if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
            raise ValueError(
                f"Slice sizes too small for n={n}, pct={self._train_pct},{self._val_pct}"
            )
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        return Split(train=train, val=val, test=test)
