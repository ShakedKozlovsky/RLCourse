"""WindowBuilder — output shape, window content, error paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqn_trader.data.window_builder import WindowBuilder


def _df(n_rows: int, n_cols: int) -> pd.DataFrame:
    return pd.DataFrame(np.arange(n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols))


def test_output_shape() -> None:
    out = WindowBuilder(30).build(_df(100, 8))
    assert out.shape == (71, 30, 8)
    assert out.dtype == np.float32


def test_first_window_equals_first_rows() -> None:
    df = _df(50, 4)
    out = WindowBuilder(10).build(df)
    np.testing.assert_array_equal(out[0], df.to_numpy(dtype=np.float32)[:10])


def test_last_window_equals_last_rows() -> None:
    df = _df(50, 4)
    out = WindowBuilder(10).build(df)
    np.testing.assert_array_equal(out[-1], df.to_numpy(dtype=np.float32)[-10:])


def test_too_short_raises() -> None:
    with pytest.raises(ValueError):
        WindowBuilder(30).build(_df(10, 4))


def test_window_size_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        WindowBuilder(1)
