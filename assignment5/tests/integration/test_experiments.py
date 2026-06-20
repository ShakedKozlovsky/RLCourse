"""Layer 11 — ExperimentService tiny smoke + aggregation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roomba_lab.sdk.experiments import ExperimentService
from roomba_lab.sdk.sdk import RoombaLab


def test_experiment_service_runs_tiny_noise_sigma_sweep(tmp_path: Path,
                                                        monkeypatch) -> None:
    lab = RoombaLab()
    # Re-route results into tmp_path
    monkeypatch.setitem(lab.config.setup, "paths",
                        {**lab.config.setup["paths"], "results_dir": str(tmp_path)})
    monkeypatch.setitem(lab.config.setup, "experiments",
                        {**lab.config.setup["experiments"],
                         "noise_sigma_sweep": [0.0, 0.2],
                         "ablation_seeds": [0]})
    svc = ExperimentService(lab, n_seeds=1, total_timesteps=300)
    out = svc.run("noise_sigma")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["kind"] == "noise_sigma"
    assert len(payload["cells"]) == 2  # 2 cells × 1 seed


def test_experiment_aggregate_returns_per_cell_means(tmp_path: Path) -> None:
    report = {
        "kind": "demo", "n_seeds": 3, "total_timesteps": 100,
        "cells": [
            {"cell": "0.0", "seed": 0, "final_reward": 10.0, "final_coverage": 0.1,
             "critic_loss_last": 1.0, "final_sigma": 0.0},
            {"cell": "0.0", "seed": 1, "final_reward": 12.0, "final_coverage": 0.11,
             "critic_loss_last": 1.0, "final_sigma": 0.0},
            {"cell": "0.5", "seed": 0, "final_reward": 30.0, "final_coverage": 0.5,
             "critic_loss_last": 0.5, "final_sigma": 0.5},
        ],
    }
    p = tmp_path / "x.json"
    p.write_text(json.dumps(report))
    agg = ExperimentService.aggregate(p)
    assert "0.0" in agg
    assert "0.5" in agg
    assert agg["0.0"]["mean_reward"] == pytest.approx(11.0)
    assert agg["0.5"]["n_seeds"] == 1.0


def test_experiment_service_rejects_unknown_kind() -> None:
    svc = ExperimentService(RoombaLab(), n_seeds=1, total_timesteps=100)
    with pytest.raises(ValueError):
        svc.run("not_a_real_sweep")
