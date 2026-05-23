"""ZScoreScaler — fit-once contract, transform shape, save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dqn_trader.data.scaler import ZScoreScaler


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])


def test_fit_transform_zero_mean_unit_std(df: pd.DataFrame) -> None:
    sc = ZScoreScaler().fit(df)
    z = sc.transform(df)
    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(ddof=0), 1.0, atol=1e-10)


def test_fit_twice_raises(df: pd.DataFrame) -> None:
    sc = ZScoreScaler().fit(df)
    with pytest.raises(RuntimeError):
        sc.fit(df)


def test_transform_before_fit_raises(df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        ZScoreScaler().transform(df)


def test_column_mismatch_raises(df: pd.DataFrame) -> None:
    sc = ZScoreScaler().fit(df)
    with pytest.raises(ValueError):
        sc.transform(df.rename(columns={"a": "x"}))


def test_save_load_round_trip(df: pd.DataFrame, tmp_path: Path) -> None:
    sc = ZScoreScaler().fit(df)
    path = tmp_path / "scaler.json"
    sc.save(path)
    sc2 = ZScoreScaler.load(path)
    pd.testing.assert_frame_equal(sc.transform(df), sc2.transform(df))


def test_no_leakage_means_match_train_only(df: pd.DataFrame) -> None:
    """Scaler fit on train slice must NOT shift when test data is later transformed."""
    train = df.iloc[:30]
    test = df.iloc[30:]
    sc = ZScoreScaler().fit(train)
    state_before = sc.state
    sc.transform(test)
    state_after = sc.state
    assert state_before == state_after  # transform must not refit
