"""End-to-end data pipeline: Kaggle CSVs → state vectors + per-day actions.

This is the only layer that touches the filesystem, so the rest of the
project can be tested offline with synthetic fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fitness_rl.data.feature_engineer import STATE_DIM, FeatureEngineer
from fitness_rl.data.kaggle_loader import KaggleLoader
from fitness_rl.data.preprocessor import Preprocessor
from fitness_rl.data.program_selector import ProgramSelector, SelectionCriteria
from fitness_rl.data.trajectory_builder import TrajectoryBuilder
from fitness_rl.shared.config import ConfigManager
from fitness_rl.shared.logger import get_logger
from fitness_rl.shared.types import Action, DailyStep, MuscleGroup

_logger = get_logger(__name__)

# Map dominant muscle group → per-day action label used for LSTM training.
_MUSCLE_TO_ACTION: dict[MuscleGroup, Action] = {
    MuscleGroup.PUSH: Action.PUSH,
    MuscleGroup.PULL: Action.PULL,
    MuscleGroup.LEGS: Action.LEGS,
    MuscleGroup.CORE: Action.LEGS,  # core/mobility folded into LEGS as least bad mapping
    MuscleGroup.CARDIO: Action.CARDIO,
    MuscleGroup.UNKNOWN: Action.REST,
}


@dataclass(frozen=True)
class PipelineOutput:
    """End-to-end pipeline result."""

    chosen_title: str
    n_weeks: int
    trajectory: list[DailyStep]
    states: np.ndarray  # shape (T, 16)
    actions: np.ndarray  # shape (T,) — per-day action labels for LSTM training


class DataService:
    """High-level facade for Layer 1 — used by SDK, training, and tests."""

    def __init__(self, config: ConfigManager, loader: KaggleLoader | None = None):
        self._cfg = config
        self._loader = loader or KaggleLoader(raw_dir=self._cfg.path("data_raw_dir"))

    def run(self) -> PipelineOutput:
        raw = self._loader.load(
            summary_name=str(self._cfg.get("data.program_summary_csv")),
            detailed_name=str(self._cfg.get("data.programs_detailed_csv")),
        )
        detailed_clean = Preprocessor().clean(raw.detailed)
        criteria = SelectionCriteria(
            equipment=str(self._cfg.get("data.equipment_filter")),
            min_program_weeks=int(self._cfg.get("data.min_program_weeks")),
            max_program_weeks=int(self._cfg.get("data.max_program_weeks")),
            min_time_per_workout=int(self._cfg.get("data.min_time_per_workout")),
            max_time_per_workout=int(self._cfg.get("data.max_time_per_workout")),
        )
        selector = ProgramSelector(criteria)
        chosen = selector.pick(raw.summary)
        n_weeks = int(chosen["program_length"])
        program_rows = selector.filter_detailed(detailed_clean, chosen["title"])
        trajectory = TrajectoryBuilder().build(program_rows, n_weeks)
        states = FeatureEngineer(n_weeks=n_weeks).transform(trajectory)
        actions = np.array(
            [_MUSCLE_TO_ACTION[s.dominant_muscle] for s in trajectory],
            dtype=np.int64,
        )
        assert states.shape == (len(trajectory), STATE_DIM)
        _logger.info("pipeline output: chosen=%r, %d days, state_dim=%d",
                     chosen["title"], len(trajectory), STATE_DIM)
        return PipelineOutput(
            chosen_title=str(chosen["title"]), n_weeks=n_weeks,
            trajectory=trajectory, states=states, actions=actions,
        )

    def save_chosen(self, output: PipelineOutput, out_path: Path) -> None:
        """Persist the chosen-program title for transparency."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        out_path.write_text(json.dumps({"chosen_title": output.chosen_title,
                                         "n_weeks": output.n_weeks,
                                         "n_days": len(output.trajectory)}, indent=2))
