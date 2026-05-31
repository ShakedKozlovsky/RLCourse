"""DataService — end-to-end pipeline on synthetic CSVs."""

from __future__ import annotations

from pathlib import Path

from fitness_rl.data.feature_engineer import STATE_DIM
from fitness_rl.services.data_service import DataService
from fitness_rl.shared.config import ConfigManager
from fitness_rl.shared.types import Action


def test_pipeline_end_to_end(minimal_config: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_config)
    out = DataService(cfg).run()
    assert out.chosen_title == "Program Match"
    assert out.n_weeks == 4
    assert len(out.trajectory) == 4 * 7
    assert out.states.shape == (4 * 7, STATE_DIM)
    assert out.actions.shape == (4 * 7,)


def test_action_values_are_in_range(minimal_config: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_config)
    out = DataService(cfg).run()
    assert out.actions.min() >= 0
    assert out.actions.max() < Action.n()


def test_save_chosen_writes_json(minimal_config: Path, tmp_path: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_config)
    out = DataService(cfg).run()
    json_path = tmp_path / "chosen.json"
    DataService(cfg).save_chosen(out, json_path)
    assert json_path.exists()
    text = json_path.read_text()
    assert "Program Match" in text
