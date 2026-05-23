"""Shared pytest fixtures for the whole suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """120 trading days of deterministic-but-non-trivial OHLCV starting at 100."""
    rng = np.random.default_rng(7)
    n = 400
    log_ret = rng.normal(0.0005, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    high = close * (1 + rng.uniform(0.0, 0.01, n))
    low = close * (1 - rng.uniform(0.0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.date_range("2020-01-02", periods=n, freq="B", name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Iterator[Path]:
    d = tmp_path / "cache"
    d.mkdir()
    yield d


@pytest.fixture
def minimal_setup_config(tmp_path: Path) -> Path:
    import json

    from dqn_trader.shared.version import __version__

    cfg = {
        "version": __version__,
        "seed": 42,
        "data": {
            "ticker": "AAPL",
            "start": "2021-01-04",
            "end": "2021-06-30",
            "interval": "1d",
            "train_pct": 0.7,
            "val_pct": 0.15,
            "window_size": 30,
            "features": 10,
        },
        "paths": {
            "data_raw_dir": "data/raw",
            "data_processed_dir": "data/processed",
            "results_dir": "results",
            "assets_dir": "assets",
        },
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def minimal_rate_limits(tmp_path: Path) -> Path:
    import json

    from dqn_trader.shared.version import __version__

    rl = {
        "version": __version__,
        "services": {
            "yfinance": {
                "requests_per_minute": 30,
                "requests_per_hour": 500,
                "concurrent_max": 1,
                "retry_after_seconds": 0,
                "max_retries": 1,
                "backoff_factor": 1.0,
            }
        },
    }
    p = tmp_path / "rate_limits.json"
    p.write_text(json.dumps(rl))
    return p
