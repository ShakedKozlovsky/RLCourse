"""Fetch raw OHLCV from yfinance with parquet cache and CSV fallback.

Behaviour matches the spec in ``docs/PRD_data_pipeline.md`` §Caching.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from dqn_trader.shared.gatekeeper import ApiGatekeeper
from dqn_trader.shared.logger import get_logger

_REQUIRED = ("Open", "High", "Low", "Close", "Volume")
_logger = get_logger(__name__)

# Type alias for the underlying downloader (yf.download). Tests inject a fake.
Downloader = Callable[..., pd.DataFrame]


class YFinanceClient:
    """Cache-aware Yahoo Finance fetcher. Never bypassed in tests."""

    def __init__(
        self,
        cache_dir: Path,
        gatekeeper: ApiGatekeeper,
        downloader: Downloader | None = None,
        csv_fallback_dir: Path | None = None,
    ):
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._gate = gatekeeper
        self._csv_fallback = csv_fallback_dir or cache_dir
        self._downloader = downloader or _default_downloader

    def fetch(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV for a ticker, using parquet cache or CSV fallback on failure."""
        cache_file = self._cache / f"{ticker}_{start}_{end}.parquet"
        if cache_file.exists():
            _logger.info("yfinance cache HIT %s", cache_file.name)
            return _coerce(pd.read_parquet(cache_file))
        try:
            raw = self._gate.execute(
                self._downloader,
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
            )
            df = _coerce(raw)
            if df.empty:
                raise ValueError(f"yfinance returned empty frame for {ticker}")
            _missing(df)
            df.to_parquet(cache_file, compression="snappy")
            _logger.info("yfinance cache MISS — wrote %s (%d rows)", cache_file.name, len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            _logger.warning("yfinance failed (%s) — trying CSV fallback", exc)
            return self._csv(ticker)

    def _csv(self, ticker: str) -> pd.DataFrame:
        csv_path = self._csv_fallback / f"{ticker}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No CSV fallback at {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = _coerce(df)
        _missing(df)
        return df


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance / CSV quirks: drop MultiIndex level, ensure float dtype."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(1)
    if df.index.name is None:
        df = df.copy()
        df.index.name = "Date"
    return df


def _missing(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")


def _default_downloader(**kwargs: object) -> pd.DataFrame:
    """Lazy import so unit tests don't need yfinance installed."""
    import yfinance as yf

    return yf.download(**kwargs)
