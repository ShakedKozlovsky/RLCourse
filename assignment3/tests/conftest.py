"""Shared pytest fixtures — synthetic CSV data so tests run offline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_summary() -> pd.DataFrame:
    """Two programs — only one matches our selection criteria."""
    return pd.DataFrame({
        "title": ["Program Match", "Program NoMatch"],
        "description": ["a", "b"],
        "level": ["['Intermediate']", "['Beginner']"],
        "goal": ["['Bodybuilding']", "['Cardio']"],
        "equipment": ["Full Gym", "None"],
        "program_length": [4.0, 2.0],
        "time_per_workout": [60.0, 30.0],
        "total_exercises": [56, 20],
        "created": ["2024-01-01", "2024-01-01"],
        "last_edit": ["2024-06-01", "2024-06-01"],
    })


@pytest.fixture
def synthetic_detailed() -> pd.DataFrame:
    """A 4-week × 4-day training program with sample exercises."""
    rows = []
    for week in range(1, 5):
        # Day 1: PUSH (bench, shoulder press)
        rows += [{"week": week, "day": 1, "exercise_name": ex, "sets": 3.0, "reps": 8.0,
                  "intensity": 7, "time_per_workout": 60.0} for ex in
                 ("Bench Press", "Shoulder Press", "Tricep Dip")]
        # Day 2: LEGS (squat, leg press)
        rows += [{"week": week, "day": 2, "exercise_name": ex, "sets": 4.0, "reps": 8.0,
                  "intensity": 8, "time_per_workout": 75.0} for ex in
                 ("Squat", "Leg Press", "Calf Raise")]
        # Day 4: PULL (row, pullup)
        rows += [{"week": week, "day": 4, "exercise_name": ex, "sets": 3.0, "reps": 10.0,
                  "intensity": 7, "time_per_workout": 60.0} for ex in
                 ("Barbell Row", "Pull-up", "Bicep Curl")]
        # Day 5: One negative-rep row (interpreted as seconds for plank)
        rows.append({"week": week, "day": 5, "exercise_name": "Plank",
                     "sets": 3.0, "reps": -60.0, "intensity": 5, "time_per_workout": 45.0})
        # Day 7: rest day (no rows)
    df = pd.DataFrame(rows)
    df.insert(0, "title", "Program Match")
    return df


@pytest.fixture
def minimal_config(tmp_path: Path, synthetic_summary: pd.DataFrame,
                   synthetic_detailed: pd.DataFrame) -> Path:
    """Drop CSVs into a tmp data/raw/ dir and write a matching config."""
    from fitness_rl.shared.version import __version__

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    synthetic_summary.to_csv(raw_dir / "program_summary.csv", index=False)
    synthetic_detailed.to_csv(raw_dir / "programs_detailed_boostcamp_kaggle.csv", index=False)
    cfg = {
        "version": __version__, "seed": 42,
        "data": {"raw_dir": "data/raw",
                 "program_summary_csv": "program_summary.csv",
                 "programs_detailed_csv": "programs_detailed_boostcamp_kaggle.csv",
                 "equipment_filter": "Full Gym",
                 "min_program_weeks": 4, "max_program_weeks": 12,
                 "min_time_per_workout": 45, "max_time_per_workout": 120},
        "paths": {"data_raw_dir": str(raw_dir), "results_dir": str(tmp_path / "results"),
                  "assets_dir": str(tmp_path / "assets"),
                  "checkpoints_dir": str(tmp_path / "saved_models")},
    }
    cfg_path = tmp_path / "setup.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run a test from a tmp working directory; restored on teardown."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def sdk_config(minimal_config: Path) -> Path:
    """``minimal_config`` extended with env / world_model / reinforce / a2c sections.

    Tiny hyperparameters so integration tests stay under a few seconds.
    """
    cfg = json.loads(minimal_config.read_text())
    cfg["env"] = {
        "state_dim": 16, "n_actions": 5,
        "episode_length": 6, "gamma": 0.99,
        "reward_gain_weight": 1.0,
        "reward_overload_lambda": 0.2,
        "reward_imbalance_lambda": 0.3,
        "action_masking_enabled": False,
        "max_same_group_consecutive": 2,
        "max_rest_consecutive": 2,
    }
    cfg["world_model"] = {
        "hidden_size": 8, "num_layers": 1, "window_size": 4,
        "epochs": 3, "batch_size": 4, "lr": 0.01,
        "early_stop_patience": 5, "train_pct": 0.8,
    }
    cfg["reinforce"] = {
        "episodes": 3, "lr": 0.01, "use_baseline": True,
        "policy_hidden": 16, "entropy_bonus": 0.0,
    }
    cfg["a2c"] = {
        "episodes": 3, "actor_lr": 0.005, "critic_lr": 0.01,
        "hidden": 16, "entropy_bonus": 0.01,
    }
    minimal_config.write_text(json.dumps(cfg))
    return minimal_config
