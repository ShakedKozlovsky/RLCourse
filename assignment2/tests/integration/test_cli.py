"""CLI smoke tests via Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from dqn_trader.interface.cli.main import cli
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.version import __version__


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    cfg = {
        "version": __version__, "seed": 1,
        "data": {"ticker": "AAPL", "start": "2021-01-04", "end": "2021-06-30",
                 "interval": "1d", "train_pct": 0.7, "val_pct": 0.15,
                 "window_size": 30, "features": 10},
        "env": {"initial_capital": 10000, "transaction_cost_alpha": 0.001,
                "slippage_beta": 0.001, "invalid_action_penalty": 0.0,
                "reward_variant": "baseline", "sharpe_bonus_gamma": 1.0, "sharpe_window": 20},
        "agent": {"gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
                  "epsilon_decay_steps": 500, "lr": 1e-3, "batch_size": 16,
                  "replay_capacity": 1000, "min_replay_size": 16,
                  "target_sync_every": 50, "huber_delta": 1.0, "grad_clip": 10.0,
                  "dueling": True, "double_dqn": True},
        "per": {"enabled": True, "alpha": 0.6, "beta_start": 0.4, "beta_end": 1.0,
                "beta_anneal_steps": 500, "epsilon": 1e-6},
        "training": {"episodes": 1, "max_steps_per_episode": None,
                     "eval_every_episodes": 1,
                     "checkpoint_dir": "results/checkpoints", "best_metric": "val_return"},
        "backtest": {"deterministic_policy": "greedy",
                     "report_dir": str(tmp_path / "results" / "backtest")},
        "paths": {"data_raw_dir": "data/raw", "data_processed_dir": "data/processed",
                  "results_dir": str(tmp_path / "results"), "assets_dir": "assets"},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output and "train" in result.output


def test_cli_data_command(
    cli_config: Path, minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DQN_TRADER_RATE_LIMITS", str(minimal_rate_limits))
    cfg = ConfigManager(setup_path=cli_config, rate_limits_path=minimal_rate_limits)
    raw_dir = cfg.path("data_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    result = CliRunner().invoke(cli, ["--config", str(cli_config), "data"])
    assert result.exit_code == 0, result.output
    assert "train features:" in result.output


def test_cli_train_then_backtest_predict(
    cli_config: Path, minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DQN_TRADER_RATE_LIMITS", str(minimal_rate_limits))
    cfg = ConfigManager(setup_path=cli_config, rate_limits_path=minimal_rate_limits)
    raw_dir = cfg.path("data_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    runner = CliRunner()

    train_r = runner.invoke(cli, ["--config", str(cli_config), "train"])
    assert train_r.exit_code == 0, train_r.output

    # locate the best.pt produced by training
    results_dir = cfg.path("results_dir")
    best_pts = sorted(results_dir.rglob("best.pt"))
    assert best_pts, "best.pt was not created"
    ckpt = best_pts[-1]

    bt_r = runner.invoke(cli, ["--config", str(cli_config), "backtest", "--checkpoint", str(ckpt)])
    assert bt_r.exit_code == 0, bt_r.output
    payload = json.loads(bt_r.output)
    assert "total_return" in payload

    pr_r = runner.invoke(cli, ["--config", str(cli_config), "predict", "--checkpoint", str(ckpt)])
    assert pr_r.exit_code == 0, pr_r.output
    p = json.loads(pr_r.output)
    assert p["action"] in {"SELL", "HOLD", "BUY"}
