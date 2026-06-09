"""Click CLI smoke tests using CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from proximal_lab.interface.cli.main import cli


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "secondary_id": "Walker2d-v5",
                 "max_episode_steps": 1000, "n_parallel_envs": 2, "gamma": 0.99},
        "gae": {"lambda": 0.95},
        "actor_critic": {"hidden_sizes": [32, 32], "activation": "tanh",
                          "shared_trunk": False,
                          "log_std_init": -0.5, "log_std_min": -5.0, "log_std_max": 2.0},
        "ppo": {"total_timesteps": 256, "steps_per_rollout": 128,
                 "minibatch_size": 16, "n_epochs_per_update": 1,
                 "clip_eps": 0.2, "lr": 3e-4, "value_coef": 0.5,
                 "entropy_coef": 0.0, "max_grad_norm": 0.5, "target_kl_stop": None},
        "paths": {"results_dir": str(tmp_path / "results"),
                   "assets_dir": str(tmp_path / "assets"),
                   "checkpoints_dir": str(tmp_path / "saved_models"),
                   "wiki_dir": str(tmp_path / "wiki")},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def test_help_lists_subcommands() -> None:
    result = CliRunner().invoke(cli, ["--help"], obj={})
    assert result.exit_code == 0
    for cmd in ("train", "evaluate", "graphify", "sweep", "gui"):
        assert cmd in result.output


def test_train_command_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, ["--config", str(sdk_config), "train"], obj={},
    )
    assert result.exit_code == 0, result.output
    assert "final_mean_reward" in result.output


def test_evaluate_command_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, ["--config", str(sdk_config), "evaluate", "--n-episodes", "1"], obj={},
    )
    assert result.exit_code == 0, result.output
    assert "mean_reward" in result.output
