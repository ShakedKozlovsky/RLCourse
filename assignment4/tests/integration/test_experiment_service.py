"""ExperimentService — λ / γ / clip-ε sweeps with 1-seed smoke configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.services.experiment_service import (
    ExperimentService,
    aggregate_seeds,
)


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "gamma": 0.99, "n_parallel_envs": 2,
                 "secondary_id": "Walker2d-v5", "max_episode_steps": 1000},
        "gae": {"lambda": 0.95},
        "actor_critic": {"hidden_sizes": [32, 32], "activation": "tanh",
                          "shared_trunk": False,
                          "log_std_init": -0.5, "log_std_min": -5.0, "log_std_max": 2.0},
        "ppo": {"total_timesteps": 256, "steps_per_rollout": 128,
                 "minibatch_size": 16, "n_epochs_per_update": 1,
                 "clip_eps": 0.2, "lr": 3e-4, "value_coef": 0.5,
                 "entropy_coef": 0.0, "max_grad_norm": 0.5, "target_kl_stop": None},
        "experiments": {"lambda_sweep": [0.0, 0.95],
                         "gamma_sweep": [0.95, 0.99],
                         "clip_eps_sweep": [0.1, 0.2]},
        "paths": {"results_dir": str(tmp_path / "results"),
                   "assets_dir": str(tmp_path / "assets"),
                   "checkpoints_dir": str(tmp_path / "saved_models"),
                   "wiki_dir": str(tmp_path / "wiki")},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def test_invalid_init_raises(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    with pytest.raises(ValueError):
        ExperimentService(sdk, timesteps_per_cell=0)
    with pytest.raises(ValueError):
        ExperimentService(sdk, n_seeds=0)


def test_lambda_sweep_returns_report(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    svc = ExperimentService(sdk, timesteps_per_cell=128, steps_per_rollout=64, n_seeds=1)
    report = svc.run_lambda_sweep()
    assert report.label == "lambda_sweep"
    assert len(report.cells) == 2  # two λ values
    for cell in report.cells:
        assert cell.n_seeds == 1
        assert "lambda=" in cell.name


def test_gamma_sweep_returns_report(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    svc = ExperimentService(sdk, timesteps_per_cell=128, steps_per_rollout=64, n_seeds=1)
    report = svc.run_gamma_sweep()
    assert report.label == "gamma_sweep"
    assert len(report.cells) == 2


def test_clip_eps_sweep_returns_report(sdk_config: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    svc = ExperimentService(sdk, timesteps_per_cell=128, steps_per_rollout=64, n_seeds=1)
    report = svc.run_clip_eps_sweep()
    assert report.label == "clip_eps_sweep"


def test_save_writes_json(sdk_config: Path, tmp_path: Path) -> None:
    sdk = ProximalLab(config_path=sdk_config)
    svc = ExperimentService(sdk, timesteps_per_cell=128, steps_per_rollout=64, n_seeds=1)
    report = svc.run_lambda_sweep()
    out = tmp_path / "lambda.json"
    ExperimentService.save(report, out)
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["label"] == "lambda_sweep"


def test_aggregate_seeds_helper() -> None:
    out = aggregate_seeds([1.0, 3.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["std"] > 0
    assert out["n_seeds"] == 2


def test_aggregate_seeds_empty() -> None:
    out = aggregate_seeds([])
    assert out["mean"] == 0.0
    assert out["std"] == 0.0
    assert out["n_seeds"] == 0
