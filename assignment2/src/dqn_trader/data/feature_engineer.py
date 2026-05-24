"""Compute the 8 market feature channels from raw OHLCV.

The 10 state channels are: these 8 + ``position``, ``pnl_unrealised`` added
by ``TradingEnv`` at runtime. See ``docs/PRD_features.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "vwap_dist",
    "volume_norm",
)


class FeatureEngineer:
    """Stateless transformer. Given a raw OHLCV DataFrame returns features only."""

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_sigma: float = 2.0,
        vwap_period: int = 20,
        volume_period: int = 60,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_sigma = bb_sigma
        self.vwap_period = vwap_period
        self.volume_period = volume_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Compute the 8 market feature channels from raw OHLCV."""
        close = raw["Close"].astype(float)
        volume = raw["Volume"].astype(float)
        out = pd.DataFrame(index=raw.index)
        out["log_return"] = np.log(close / close.shift(1))
        out["rsi_14"] = _rsi(close, self.rsi_period)
        macd = _ema(close, self.macd_fast) - _ema(close, self.macd_slow)
        signal = _ema(macd, self.macd_signal)
        out["macd"] = macd
        out["macd_signal"] = signal
        out["macd_hist"] = macd - signal
        out["bb_pct"] = _bb_pct(close, self.bb_period, self.bb_sigma)
        out["vwap_dist"] = _vwap_dist(close, volume, self.vwap_period)
        out["volume_norm"] = _zscore(np.log(volume.clip(lower=1.0)), self.volume_period)
        return out[list(FEATURE_COLUMNS)].dropna()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI on a Close series."""
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _bb_pct(close: pd.Series, period: int, sigma: float) -> pd.Series:
    """Bollinger %B in the natural [0, 1] band (can exceed when price breaks out)."""
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + sigma * sd
    lower = ma - sigma * sd
    return (close - lower) / (upper - lower)


def _vwap_dist(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    pv = (close * volume).rolling(period).sum()
    v = volume.rolling(period).sum()
    vwap = pv / v
    return (close - vwap) / vwap


def _zscore(s: pd.Series, period: int) -> pd.Series:
    mu = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd
