"""ExperimentService — integration smoke on synthetic OHLCV for one experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dqn_trader.services.experiment_service import ExperimentService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.version import __version__


@pytest.fixture
def exp_config(tmp_path: Path, synthetic_ohlcv: pd.DataFrame) -> tuple[Path, Path]:
    cfg = {
        "version": __version__, "seed": 1,
        "data": {"ticker": "AAPL", "start": "2021-01-04", "end": "2021-06-30",
                 "interval": "1d", "train_pct": 0.7, "val_pct": 0.15,
                 "window_size": 30, "features": 10, "comparative_ticker": "AAPL"},
        "env": {"initial_capital": 10000, "transaction_cost_alpha": 0.001,
                "slippage_beta": 0.001, "invalid_action_penalty": 0.0,
                "reward_variant": "baseline", "sharpe_bonus_gamma": 1.0, "sharpe_window": 20},
        "agent": {"gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
                  "epsilon_decay_steps": 200, "lr": 1e-3, "batch_size": 8,
                  "replay_capacity": 200, "min_replay_size": 8,
                  "target_sync_every": 50, "huber_delta": 1.0, "grad_clip": 10.0,
                  "dueling": True, "double_dqn": True},
        "per": {"enabled": True, "alpha": 0.6, "beta_start": 0.4, "beta_end": 1.0,
                "beta_anneal_steps": 200, "epsilon": 1e-6},
        "training": {"episodes": 1, "max_steps_per_episode": None,
                     "eval_every_episodes": 1,
                     "checkpoint_dir": "results/checkpoints", "best_metric": "val_return"},
        "backtest": {"deterministic_policy": "greedy",
                     "report_dir": str(tmp_path / "results" / "backtest")},
        "paths": {"data_raw_dir": str(tmp_path / "data" / "raw"),
                  "data_processed_dir": str(tmp_path / "data" / "processed"),
                  "results_dir": str(tmp_path / "results"), "assets_dir": "assets"},
    }
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps(cfg))
    raw_dir = Path(cfg["paths"]["data_raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    return setup_path, tmp_path


def test_run_dqn_vs_dueling(exp_config: tuple[Path, Path], minimal_rate_limits: Path) -> None:
    setup_path, tmp_path = exp_config
    cfg = ConfigManager(setup_path=setup_path, rate_limits_path=minimal_rate_limits)
    exp = ExperimentService(cfg)
    res = exp.run_dqn_vs_dueling()
    assert res.name == "dqn_vs_dueling"
    assert len(res.conditions) == 2
    cond_names = {c.name for c in res.conditions}
    assert {"vanilla_dqn", "dueling_dqn"} == cond_names
    summary_md = tmp_path / "results" / "experiments_summary.md"
    assert summary_md.exists() and "dqn_vs_dueling" in summary_md.read_text()
    json_path = tmp_path / "results" / "dqn_vs_dueling.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["name"] == "dqn_vs_dueling"
    assert len(payload["conditions"]) == 2


def test_run_uniform_vs_per(exp_config: tuple[Path, Path], minimal_rate_limits: Path) -> None:
    setup_path, _ = exp_config
    cfg = ConfigManager(setup_path=setup_path, rate_limits_path=minimal_rate_limits)
    res = ExperimentService(cfg).run_uniform_vs_per()
    assert {c.name for c in res.conditions} == {"uniform_replay", "prioritized_replay"}


def test_run_reward_variants(exp_config: tuple[Path, Path], minimal_rate_limits: Path) -> None:
    setup_path, _ = exp_config
    cfg = ConfigManager(setup_path=setup_path, rate_limits_path=minimal_rate_limits)
    res = ExperimentService(cfg).run_reward_variants()
    assert {c.name for c in res.conditions} == {"baseline", "risk_adjusted"}
