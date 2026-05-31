"""Load the two Kaggle CSVs into pandas DataFrames.

The detailed CSV is ~280 MB and has ~600k rows, so we use ``low_memory=False``
to suppress dtype warnings on mixed columns. The loader does *no* cleaning —
that's the next layer (preprocessor).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitness_rl.shared.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class RawDataset:
    """The two CSVs loaded as DataFrames, with no cleaning yet."""

    summary: pd.DataFrame
    detailed: pd.DataFrame


class KaggleLoader:
    """Reads ``program_summary.csv`` and ``programs_detailed_boostcamp_kaggle.csv``."""

    def __init__(self, raw_dir: Path):
        self._raw_dir = raw_dir

    def load(
        self,
        summary_name: str = "program_summary.csv",
        detailed_name: str = "programs_detailed_boostcamp_kaggle.csv",
    ) -> RawDataset:
        """Read both CSVs; raise FileNotFoundError if either is missing."""
        summary_path = self._raw_dir / summary_name
        detailed_path = self._raw_dir / detailed_name
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
        if not detailed_path.exists():
            raise FileNotFoundError(f"Missing detailed CSV: {detailed_path}")
        _logger.info("loading %s", summary_path)
        summary = pd.read_csv(summary_path)
        _logger.info("loading %s (large file, may take a moment)", detailed_path)
        detailed = pd.read_csv(detailed_path, low_memory=False)
        _logger.info(
            "loaded: %d programs, %d detailed rows", len(summary), len(detailed)
        )
        return RawDataset(summary=summary, detailed=detailed)
