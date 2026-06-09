"""ConfigManager — JSON loader + dotted access + version check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proximal_lab.shared.config import ConfigError, ConfigManager


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_returns_dotted_values(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, {
        "version": "1.00", "env": {"id": "HalfCheetah-v5", "gamma": 0.99},
    })
    cfg = ConfigManager(setup_path=cfg_path)
    assert cfg.get("env.id") == "HalfCheetah-v5"
    assert cfg.get("env.gamma") == pytest.approx(0.99)


def test_missing_key_returns_default(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, {"version": "1.00", "env": {"gamma": 0.99}})
    cfg = ConfigManager(setup_path=cfg_path)
    assert cfg.get("env.id", "fallback") == "fallback"
    assert cfg.get("nothing.here.deep") is None


def test_version_mismatch_raises(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, {"version": "0.99"})
    with pytest.raises(ConfigError):
        ConfigManager(setup_path=cfg_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ConfigManager(setup_path=tmp_path / "does-not-exist.json")


def test_path_resolves_against_project_root(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, {
        "version": "1.00", "paths": {"results_dir": "results"},
    })
    cfg = ConfigManager(setup_path=cfg_path)
    p = cfg.path("results_dir")
    assert p.name == "results"


def test_path_missing_raises(tmp_path: Path) -> None:
    cfg_path = _write(tmp_path, {"version": "1.00"})
    cfg = ConfigManager(setup_path=cfg_path)
    with pytest.raises(ConfigError):
        cfg.path("results_dir")
