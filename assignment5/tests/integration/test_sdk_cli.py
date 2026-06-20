"""Layer 9 — SDK + CLI smoke."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from roomba_lab.interface.cli.main import cli
from roomba_lab.sdk.sdk import RoombaLab


def test_sdk_make_env_returns_env() -> None:
    lab = RoombaLab()
    env = lab.make_env()
    assert env.obs_dim > 0
    assert env.action_dim == 2


def test_sdk_train_short_run() -> None:
    lab = RoombaLab()
    result = lab.train(total_timesteps=400, seed=0)
    assert len(result.diagnostics) > 0


def test_cli_help() -> None:
    runner = CliRunner()
    out = runner.invoke(cli, ["--help"])
    assert out.exit_code == 0
    assert "train" in out.output
    assert "graphify" in out.output


def test_cli_train_short_run(tmp_path: Path) -> None:
    save = tmp_path / "diag.json"
    runner = CliRunner()
    out = runner.invoke(cli, ["train", "--total-timesteps", "300", "--save", str(save)])
    assert out.exit_code == 0, out.output
    assert save.exists()
    payload = json.loads(save.read_text())
    assert "diagnostics" in payload


def test_cli_download_data_idempotent() -> None:
    runner = CliRunner()
    out = runner.invoke(cli, ["download-data"])
    assert out.exit_code == 0


def test_cli_help_lists_six_subcommands() -> None:
    runner = CliRunner()
    out = runner.invoke(cli, ["--help"])
    expected = ["train", "evaluate", "download-data", "record-gif", "sweep", "graphify", "gui"]
    for name in expected:
        assert name in out.output
