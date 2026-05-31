"""Click CLI smoke tests using CliRunner."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from fitness_rl.interface.cli.main import cli


def _runner_args(config: Path, *subcommand: str) -> list[str]:
    return ["--config", str(config), *subcommand]


def test_help_shows_subcommands() -> None:
    result = CliRunner().invoke(cli, ["--help"], obj={})
    assert result.exit_code == 0
    for cmd in ("prepare-data", "train-world", "train-reinforce", "train-a2c",
                "compare", "predict", "menu"):
        assert cmd in result.output


def test_prepare_data_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(cli, _runner_args(sdk_config, "prepare-data"), obj={})
    assert result.exit_code == 0, result.output
    assert "Program Match" in result.output


def test_train_reinforce_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, _runner_args(sdk_config, "train-reinforce", "--episodes", "2"), obj={}
    )
    assert result.exit_code == 0, result.output
    assert "episodes=2" in result.output


def test_train_a2c_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, _runner_args(sdk_config, "train-a2c", "--episodes", "2"), obj={}
    )
    assert result.exit_code == 0, result.output
    assert "episodes=2" in result.output


def test_compare_writes_json(sdk_config: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "comparison.json"
    result = CliRunner().invoke(
        cli,
        _runner_args(sdk_config, "compare", "--episodes", "2", "--out", str(out_json)),
        obj={},
    )
    assert result.exit_code == 0, result.output
    assert "winner=" in result.output
    assert out_json.exists()


def test_predict_runs(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, _runner_args(sdk_config, "predict", "--algo", "a2c", "--episodes", "2"),
        obj={},
    )
    assert result.exit_code == 0, result.output
    assert "action=" in result.output


def test_menu_quits_on_q(sdk_config: Path) -> None:
    result = CliRunner().invoke(
        cli, _runner_args(sdk_config, "menu"), input="q\n", obj={},
    )
    assert result.exit_code == 0
