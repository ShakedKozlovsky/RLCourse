"""ConfigManager — version check, dotted access, path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dqn_trader.shared.config import ConfigError, ConfigManager


def test_loads_and_exposes_dotted_keys(
    minimal_setup_config: Path, minimal_rate_limits: Path
) -> None:
    cfg = ConfigManager(setup_path=minimal_setup_config, rate_limits_path=minimal_rate_limits)
    assert cfg.get("data.ticker") == "AAPL"
    assert cfg.get("data.window_size") == 30
    assert cfg.get("data.missing", default="x") == "x"


def test_version_mismatch_raises(tmp_path: Path, minimal_rate_limits: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": "0.99", "paths": {}}))
    with pytest.raises(ConfigError, match="Version mismatch"):
        ConfigManager(setup_path=bad, rate_limits_path=minimal_rate_limits)


def test_missing_file_raises(tmp_path: Path, minimal_rate_limits: Path) -> None:
    with pytest.raises(ConfigError, match="Config not found"):
        ConfigManager(setup_path=tmp_path / "no.json", rate_limits_path=minimal_rate_limits)


def test_path_resolution(minimal_setup_config: Path, minimal_rate_limits: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_setup_config, rate_limits_path=minimal_rate_limits)
    p = cfg.path("data_raw_dir")
    assert p.name == "raw"
    assert p.parent.name == "data"


def test_path_missing_raises(minimal_setup_config: Path, minimal_rate_limits: Path) -> None:
    cfg = ConfigManager(setup_path=minimal_setup_config, rate_limits_path=minimal_rate_limits)
    with pytest.raises(ConfigError, match="missing"):
        cfg.path("does_not_exist")
