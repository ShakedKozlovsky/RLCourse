"""YFinanceClient — cache hit, fallback, MultiIndex coercion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dqn_trader.data.yfinance_client import YFinanceClient
from dqn_trader.shared.gatekeeper import ApiGatekeeper

_CFG = {
    "requests_per_minute": 30,
    "requests_per_hour": 500,
    "retry_after_seconds": 0,
    "max_retries": 1,
    "backoff_factor": 1.0,
}


@pytest.fixture
def gate() -> ApiGatekeeper:
    return ApiGatekeeper(_CFG, clock=lambda: 0.0)


def test_cache_hit_does_not_call_downloader(
    tmp_cache: Path, gate: ApiGatekeeper, synthetic_ohlcv: pd.DataFrame
) -> None:
    synthetic_ohlcv.to_parquet(
        tmp_cache / "AAPL_2021-01-04_2021-06-30.parquet", compression="snappy"
    )
    calls = {"n": 0}

    def downloader(**_: object) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()

    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=downloader)
    df = client.fetch("AAPL", "2021-01-04", "2021-06-30")
    assert calls["n"] == 0
    assert "Close" in df.columns


def test_cache_miss_writes_parquet(
    tmp_cache: Path, gate: ApiGatekeeper, synthetic_ohlcv: pd.DataFrame
) -> None:
    def downloader(**_: object) -> pd.DataFrame:
        return synthetic_ohlcv

    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=downloader)
    client.fetch("AAPL", "2021-01-04", "2021-06-30")
    assert (tmp_cache / "AAPL_2021-01-04_2021-06-30.parquet").exists()


def test_multiindex_columns_are_collapsed(
    tmp_cache: Path, gate: ApiGatekeeper, synthetic_ohlcv: pd.DataFrame
) -> None:
    raw = synthetic_ohlcv.copy()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])

    def downloader(**_: object) -> pd.DataFrame:
        return raw

    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=downloader)
    df = client.fetch("AAPL", "2021-01-04", "2021-06-30")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_csv_fallback_on_downloader_failure(
    tmp_cache: Path, gate: ApiGatekeeper, synthetic_ohlcv: pd.DataFrame
) -> None:
    def boom(**_: object) -> pd.DataFrame:
        raise RuntimeError("network down")

    synthetic_ohlcv.to_csv(tmp_cache / "AAPL.csv")
    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=boom)
    df = client.fetch("AAPL", "2021-01-04", "2021-06-30")
    assert "Close" in df.columns


def test_missing_columns_raises(
    tmp_cache: Path, gate: ApiGatekeeper, synthetic_ohlcv: pd.DataFrame
) -> None:
    bad = synthetic_ohlcv.drop(columns=["Volume"])

    def downloader(**_: object) -> pd.DataFrame:
        return bad

    client = YFinanceClient(cache_dir=tmp_cache, gatekeeper=gate, downloader=downloader)
    # The client falls back to CSV after the missing-columns error; absence raises.
    with pytest.raises(FileNotFoundError):
        client.fetch("AAPL", "2021-01-04", "2021-06-30")
