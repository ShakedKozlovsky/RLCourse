"""ChronologicalSplitter — exact ratios, no shuffling, monotonic time."""

from __future__ import annotations

import pandas as pd
import pytest

from dqn_trader.data.splitter import ChronologicalSplitter


def test_exact_ratios(synthetic_ohlcv: pd.DataFrame) -> None:
    s = ChronologicalSplitter(0.7, 0.15).split(synthetic_ohlcv)
    n = len(synthetic_ohlcv)
    assert len(s.train) == round(n * 0.7)
    assert len(s.val) == round(n * 0.15)
    assert len(s.test) == n - len(s.train) - len(s.val)


def test_chronological_order_preserved(synthetic_ohlcv: pd.DataFrame) -> None:
    s = ChronologicalSplitter(0.7, 0.15).split(synthetic_ohlcv)
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()


def test_invalid_ratios_raise() -> None:
    with pytest.raises(ValueError):
        ChronologicalSplitter(0.0, 0.5)
    with pytest.raises(ValueError):
        ChronologicalSplitter(0.6, 0.6)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        ChronologicalSplitter(0.7, 0.15).split(pd.DataFrame())
