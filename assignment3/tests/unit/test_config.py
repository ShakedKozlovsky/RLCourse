"""ConfigManager — version check, dotted access, path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fitness_rl.shared.config import ConfigError, ConfigManager


def test_loads_and_exposes_dotted_keys(minimal_config: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_config)
    assert cfg.get("data.equipment_filter") == "Full Gym"
    assert cfg.get("data.min_program_weeks") == 4
    assert cfg.get("data.missing", default="x") == "x"


def test_version_mismatch_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "0.99", "paths": {}}))
    with pytest.raises(ConfigError, match="Version mismatch"):
        ConfigManager(setup_path=bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config not found"):
        ConfigManager(setup_path=tmp_path / "no.json")


def test_path_resolution(minimal_config: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_config)
    p = cfg.path("data_raw_dir")
    assert p.exists()
