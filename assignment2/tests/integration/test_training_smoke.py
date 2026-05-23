"""TrainingService — end-to-end smoke test on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dqn_trader.data.yfinance_client import YFinanceClient
from dqn_trader.services.data_service import DataService
from dqn_trader.services.training_service import TrainingService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.gatekeeper import ApiGatekeeper
from dqn_trader.shared.version import __version__


@pytest.fixture
def fast_config(tmp_path: Path) -> Path:
    """A version-aligned tiny config that trains for 2 episodes."""
    cfg = {
        "version": __version__,
        "seed": 1,
        "data": {"ticker": "AAPL", "start": "2021-01-04", "end": "2021-06-30",
                 "interval": "1d", "train_pct": 0.7, "val_pct": 0.15,
                 "window_size": 30, "features": 10},
        "env": {
            "initial_capital": 10000, "transaction_cost_alpha": 0.001, "slippage_beta": 0.001,
            "invalid_action_penalty": 0.0, "reward_variant": "baseline",
            "sharpe_bonus_gamma": 1.0, "sharpe_window": 20,
        },
        "agent": {
            "gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
            "epsilon_decay_steps": 500, "lr": 0.001, "batch_size": 16,
            "replay_capacity": 1000, "min_replay_size": 16,
            "target_sync_every": 50, "huber_delta": 1.0, "grad_clip": 10.0,
            "dueling": True, "double_dqn": True,
        },
        "per": {"enabled": True, "alpha": 0.6, "beta_start": 0.4, "beta_end": 1.0,
                "beta_anneal_steps": 500, "epsilon": 1e-6},
        "training": {"episodes": 2, "max_steps_per_episode": None,
                     "eval_every_episodes": 1, "checkpoint_dir": "results/checkpoints",
                     "best_metric": "val_return"},
        "paths": {"data_raw_dir": "data/raw", "data_processed_dir": "data/processed",
                  "results_dir": str(tmp_path / "results"), "assets_dir": "assets"},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(cfg))
    return p


def test_training_runs_two_episodes_and_writes_artifacts(
    fast_config: Path, minimal_rate_limits: Path, synthetic_ohlcv: pd.DataFrame, tmp_cache: Path
) -> None:
    cfg = ConfigManager(setup_path=fast_config, rate_limits_path=minimal_rate_limits)
    gate = ApiGatekeeper(cfg.rate_limits["services"]["yfinance"], clock=lambda: 0.0)
    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=lambda **_: synthetic_ohlcv)
    pipeline = DataService(cfg, client=client).run("AAPL")
    service = TrainingService(cfg, pipeline)
    metrics, run = service.fit()
    assert len(metrics) == 2
    assert (run.checkpoints / "best.pt").exists()
    assert (run.checkpoints / "last.pt").exists()
    assert run.metrics_csv.exists()
    text = run.metrics_csv.read_text()
    assert text.startswith("episode,reward,loss,epsilon,trades,final_value,val_return")
    assert len(text.strip().split("\n")) == 3  # header + 2 episodes


def test_training_with_uniform_replay(
    fast_config: Path, minimal_rate_limits: Path, synthetic_ohlcv: pd.DataFrame, tmp_cache: Path
) -> None:
    """Same smoke test with PER disabled — exercises UniformReplay path in the service."""
    raw = json.loads(fast_config.read_text())
    raw["per"]["enabled"] = False
    fast_config.write_text(json.dumps(raw))
    cfg = ConfigManager(setup_path=fast_config, rate_limits_path=minimal_rate_limits)
    gate = ApiGatekeeper(cfg.rate_limits["services"]["yfinance"], clock=lambda: 0.0)
    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=lambda **_: synthetic_ohlcv)
    pipeline = DataService(cfg, client=client).run("AAPL")
    metrics, _ = TrainingService(cfg, pipeline).fit()
    assert len(metrics) == 2
    assert np.isfinite(metrics[-1].reward)
