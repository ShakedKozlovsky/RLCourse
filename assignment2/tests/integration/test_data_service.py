"""DataService end-to-end with an injected fake yfinance downloader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dqn_trader.data.feature_engineer import FEATURE_COLUMNS
from dqn_trader.data.yfinance_client import YFinanceClient
from dqn_trader.services.data_service import DataService
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.gatekeeper import ApiGatekeeper


def test_pipeline_runs_and_shapes_make_sense(
    minimal_setup_config: Path,
    minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame,
    tmp_cache: Path,
) -> None:
    cfg = ConfigManager(setup_path=minimal_setup_config, rate_limits_path=minimal_rate_limits)
    gate = ApiGatekeeper(cfg.rate_limits["services"]["yfinance"], clock=lambda: 0.0)
    client = YFinanceClient(
        cache_dir=tmp_cache,
        gatekeeper=gate,
        downloader=lambda **_: synthetic_ohlcv,
    )
    out = DataService(cfg, client=client).run("AAPL")
    n_feat = len(FEATURE_COLUMNS)
    win = int(cfg.get("data.window_size"))
    for name, slc in (("train", out.train), ("val", out.val), ("test", out.test)):
        assert slc.features.ndim == 3, name
        assert slc.features.shape[1:] == (win, n_feat), name
        assert slc.features.shape[0] >= 1, name
    assert out.feature_columns == FEATURE_COLUMNS


def test_pipeline_uses_cache_when_present(
    minimal_setup_config: Path,
    minimal_rate_limits: Path,
    synthetic_ohlcv: pd.DataFrame,
    tmp_cache: Path,
) -> None:
    cfg = ConfigManager(setup_path=minimal_setup_config, rate_limits_path=minimal_rate_limits)
    gate = ApiGatekeeper(cfg.rate_limits["services"]["yfinance"], clock=lambda: 0.0)
    synthetic_ohlcv.to_parquet(
        tmp_cache / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy"
    )
    calls = {"n": 0}

    def downloader(**_: object) -> pd.DataFrame:
        calls["n"] += 1
        return synthetic_ohlcv

    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=downloader)
    DataService(cfg, client=client).run("AAPL")
    assert calls["n"] == 0
