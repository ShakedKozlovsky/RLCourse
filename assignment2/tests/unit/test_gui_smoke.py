"""Headless smoke test: GUI window instantiates and shows the four tabs.

We use Qt's ``offscreen`` platform plugin so the test runs without a display
(CI, headless servers, the assignment grading box).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from dqn_trader.shared.version import __version__

# Force-offscreen BEFORE we import PyQt6 (which happens lazily below).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def gui_setup_config(tmp_path: Path) -> Path:
    cfg = {
        "version": __version__, "seed": 1,
        "data": {"ticker": "AAPL", "start": "2021-01-04", "end": "2021-06-30",
                 "interval": "1d", "train_pct": 0.7, "val_pct": 0.15,
                 "window_size": 30, "features": 10},
        "env": {"initial_capital": 10000, "transaction_cost_alpha": 0.001,
                "slippage_beta": 0.001, "invalid_action_penalty": 0.0,
                "reward_variant": "baseline", "sharpe_bonus_gamma": 1.0, "sharpe_window": 20},
        "agent": {"gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
                  "epsilon_decay_steps": 100, "lr": 1e-3, "batch_size": 8,
                  "replay_capacity": 500, "min_replay_size": 8,
                  "target_sync_every": 50, "huber_delta": 1.0, "grad_clip": 10.0,
                  "dueling": True, "double_dqn": True},
        "per": {"enabled": True, "alpha": 0.6, "beta_start": 0.4, "beta_end": 1.0,
                "beta_anneal_steps": 100, "epsilon": 1e-6},
        "training": {"episodes": 1, "max_steps_per_episode": None,
                     "eval_every_episodes": 1,
                     "checkpoint_dir": "results/checkpoints", "best_metric": "val_return"},
        "backtest": {"deterministic_policy": "greedy",
                     "report_dir": str(tmp_path / "results" / "backtest")},
        "paths": {"data_raw_dir": "data/raw", "data_processed_dir": "data/processed",
                  "results_dir": str(tmp_path / "results"), "assets_dir": "assets"},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(cfg))
    return p


def test_main_window_instantiates_with_four_tabs(
    gui_setup_config: Path, minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    from dqn_trader.interface.gui.main_window import MainWindow  # noqa: PLC0415
    from dqn_trader.sdk.sdk import TradingSDK  # noqa: PLC0415
    from dqn_trader.shared.config import ConfigManager  # noqa: PLC0415

    cfg = ConfigManager(setup_path=gui_setup_config, rate_limits_path=minimal_rate_limits)
    raw_dir = cfg.path("data_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    _ = QApplication.instance() or QApplication([])
    win = MainWindow(TradingSDK(cfg))
    assert win.tabs.count() == 4
    assert win.tabs.tabText(0) == "Data"
    assert win.tabs.tabText(1) == "Train"
    assert win.tabs.tabText(2) == "Backtest"
    assert win.tabs.tabText(3) == "Predict"
    win.close()


def test_data_tab_runs_pipeline(
    gui_setup_config: Path, minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    from dqn_trader.interface.gui.data_tab import DataTab  # noqa: PLC0415
    from dqn_trader.sdk.sdk import TradingSDK  # noqa: PLC0415
    from dqn_trader.shared.config import ConfigManager  # noqa: PLC0415

    cfg = ConfigManager(setup_path=gui_setup_config, rate_limits_path=minimal_rate_limits)
    raw_dir = cfg.path("data_raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_ohlcv.to_parquet(raw_dir / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy")
    _ = QApplication.instance() or QApplication([])
    tab = DataTab(TradingSDK(cfg))
    tab._on_prepare()  # synchronous SDK call
    text = tab._output.toPlainText()
    assert "train:" in text and "val" in text and "test" in text
