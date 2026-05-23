"""TradingSDK — end-to-end data → train → backtest → predict on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dqn_trader.data.yfinance_client import YFinanceClient
from dqn_trader.sdk.sdk import TradingSDK
from dqn_trader.services.data_service import DataService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.gatekeeper import ApiGatekeeper
from dqn_trader.shared.version import __version__


@pytest.fixture
def fast_setup_config(tmp_path: Path) -> Path:
    cfg = {
        "version": __version__,
        "seed": 1,
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
        "training": {"episodes": 2, "max_steps_per_episode": None,
                     "eval_every_episodes": 1,
                     "checkpoint_dir": "results/checkpoints",
                     "best_metric": "val_return"},
        "backtest": {"deterministic_policy": "greedy",
                     "report_dir": str(tmp_path / "results" / "backtest")},
        "paths": {"data_raw_dir": "data/raw", "data_processed_dir": "data/processed",
                  "results_dir": str(tmp_path / "results"), "assets_dir": "assets"},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def _sdk_with_fake_yf(
    setup_path: Path, rate_limits: Path, synthetic_ohlcv: pd.DataFrame, tmp_cache: Path
) -> TradingSDK:
    cfg = ConfigManager(setup_path=setup_path, rate_limits_path=rate_limits)
    sdk = TradingSDK(cfg)
    gate = ApiGatekeeper(cfg.rate_limits["services"]["yfinance"], clock=lambda: 0.0)
    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate,
                            downloader=lambda **_: synthetic_ohlcv)
    # Pre-warm the data cache via the real DataService so prepare_data hits parquet.
    DataService(cfg, client=client).run("AAPL")
    return sdk


def test_sdk_end_to_end(
    fast_setup_config: Path, minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame, tmp_cache: Path
) -> None:
    # Wire the cache to the SDK's expected raw_dir
    cfg = ConfigManager(setup_path=fast_setup_config, rate_limits_path=minimal_rate_limits)
    raw_dir = cfg.path("data_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    sdk = TradingSDK(cfg)

    train_out = sdk.train()
    assert len(train_out.metrics) == 2
    best = train_out.run_dir / "checkpoints" / "best.pt"
    assert best.exists()

    bt = sdk.backtest(best, slice_name="test", pipeline=train_out.pipeline)
    assert bt.equity.size > 0
    assert (cfg.path("results_dir") / "backtest" / "test_backtest.json").exists()

    # Predict: feed the last test window
    last_window = train_out.pipeline.test.features[-1]
    decision = sdk.predict(last_window, checkpoint=best)
    assert decision.q_values.shape == (3,)
    assert 0.0 < decision.confidence <= 1.0

    # Backtest with custom report_name writes to that filename, not the default.
    sdk.backtest(best, slice_name="test", pipeline=train_out.pipeline, report_name="custom_run")
    assert (cfg.path("results_dir") / "backtest" / "custom_run.json").exists()
