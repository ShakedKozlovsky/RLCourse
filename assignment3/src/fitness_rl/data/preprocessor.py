"""Clean the raw Kaggle DataFrames.

Per assignment §7.2.3, some negative values in sets/reps actually represent
seconds (time-based exercises like planks or stretches). We take the absolute
value when the magnitude is plausibly a duration (< 600 seconds), otherwise
drop the row.
"""

from __future__ import annotations

import pandas as pd

from fitness_rl.shared.logger import get_logger

_MAX_SECONDS = 600
_logger = get_logger(__name__)


class Preprocessor:
    """Numeric cleanup + type coercion. Stateless — same input → same output."""

    def clean(self, detailed: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the detailed DataFrame."""
        df = detailed.copy()
        for col in ("sets", "reps"):
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(self._absorb_negative)
        before = len(df)
        df = df.dropna(subset=["sets", "reps", "week", "day", "exercise_name"])
        df = df.astype({"sets": "float64", "reps": "float64", "week": "int64", "day": "int64"})
        _logger.info("preprocessor: dropped %d rows with NaN/invalid", before - len(df))
        return df.reset_index(drop=True)

    @staticmethod
    def _absorb_negative(value: float) -> float:
        """Negative numeric values represent seconds; take abs if plausible, else NaN."""
        if pd.isna(value):
            return float("nan")
        if value >= 0:
            return float(value)
        magnitude = abs(value)
        return float(magnitude) if magnitude <= _MAX_SECONDS else float("nan")
