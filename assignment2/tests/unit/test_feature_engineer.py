"""FeatureEngineer — column order, finiteness, RSI bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dqn_trader.data.feature_engineer import FEATURE_COLUMNS, FeatureEngineer


def test_output_columns_match_contract(synthetic_ohlcv: pd.DataFrame) -> None:
    out = FeatureEngineer().transform(synthetic_ohlcv)
    assert tuple(out.columns) == FEATURE_COLUMNS


def test_no_nans_after_warmup(synthetic_ohlcv: pd.DataFrame) -> None:
    out = FeatureEngineer().transform(synthetic_ohlcv)
    assert out.notna().all().all()


def test_rsi_in_0_100(synthetic_ohlcv: pd.DataFrame) -> None:
    out = FeatureEngineer().transform(synthetic_ohlcv)
    assert (out["rsi_14"] >= 0).all() and (out["rsi_14"] <= 100).all()


def test_log_return_matches_manual(synthetic_ohlcv: pd.DataFrame) -> None:
    out = FeatureEngineer().transform(synthetic_ohlcv)
    close = synthetic_ohlcv["Close"]
    expected = np.log(close / close.shift(1)).reindex(out.index)
    np.testing.assert_allclose(out["log_return"].to_numpy(), expected.to_numpy(), rtol=1e-6)


def test_warmup_truncates_rows(synthetic_ohlcv: pd.DataFrame) -> None:
    out = FeatureEngineer().transform(synthetic_ohlcv)
    # 26-period MACD slow EMA + 20-period BB warmup ⇒ first ~26 rows dropped.
    assert len(out) < len(synthetic_ohlcv)
    assert len(out) >= len(synthetic_ohlcv) - 60
