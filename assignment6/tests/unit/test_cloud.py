"""Layer 22 — cloud (local + prefect fallback) tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marl_lab.cloud.local import run_local_flow
from marl_lab.cloud.prefect import run_prefect_flow
from marl_lab.shared.types import StudentEntry


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.19", "seed": 0, "device": "cpu",
        "game": {"grid_size": [4, 4], "max_moves": 8, "num_games": 2,
                  "max_barriers": 2, "enable_barriers": False, "observation_radius": 1},
        "scoring": {"cop_win": 20, "thief_win": 10, "cop_loss": 5, "thief_loss": 5},
        "marl": {
            "algorithm": "qmix", "gamma": 0.99, "tau": 0.005, "critic_lr": 1e-3,
            "batch_size": 4, "replay_capacity": 32, "warmup_steps": 25,
            "max_grad_norm": 1.0, "hidden_sizes": [16], "rnn_hidden_size": 8,
            "embed_dim": 8, "hyper_hidden": 16,
        },
        "exploration": {"epsilon_initial": 1.0, "epsilon_final": 0.05, "decay_steps": 100},
        "training": {"total_episodes": 3},
        "experiments": {},
        "mcp": {}, "gmail": {}, "submission": {
            "group_code": "X", "group_name": "Y", "github_repo": "r",
            "timezone": "UTC",
        },
        "paths": {}, "graphify": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_local_flow_returns_report(tiny_cfg: Path) -> None:
    students = [StudentEntry(role="A", full_name="Shaked", id="1")]
    report = run_local_flow(tiny_cfg, students=students, n_episodes=2)
    assert len(report.sub_games) == 2
    assert report.group_code == "X"


def test_run_prefect_flow_falls_back_without_key(tiny_cfg: Path, monkeypatch) -> None:
    """No PREFECT_API_KEY → local fallback. Must return a valid GameReport."""
    monkeypatch.delenv("PREFECT_API_KEY", raising=False)
    students = [StudentEntry(role="A", full_name="Shaked", id="1")]
    report = run_prefect_flow(tiny_cfg, students=students, n_episodes=2)
    assert len(report.sub_games) == 2
