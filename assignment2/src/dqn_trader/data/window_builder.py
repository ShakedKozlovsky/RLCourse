"""Build rolling (N, window, features) tensors from a per-day feature frame."""

from __future__ import annotations

import numpy as np
import pandas as pd


class WindowBuilder:
    """Stride-1 rolling windows over the time axis. Output dtype float32."""

    def __init__(self, window_size: int):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size

    def build(self, scaled: pd.DataFrame) -> np.ndarray:
        """Return tensor of shape ``(N, window_size, n_features)``.

        ``N = len(scaled) - window_size + 1``. The output row at index ``i``
        corresponds to days ``[i .. i + window_size - 1]`` of ``scaled``.
        """
        if len(scaled) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} rows, got {len(scaled)}")
        arr = scaled.to_numpy(dtype=np.float32)
        n_feat = arr.shape[1]
        # np.lib.stride_tricks gives an O(1)-memory view; we copy to own the buffer.
        windows = np.lib.stride_tricks.sliding_window_view(
            arr, window_shape=(self.window_size, n_feat)
        )
        # sliding_window_view returns shape (N, 1, window, feat); squeeze the dummy axis.
        return np.ascontiguousarray(windows[:, 0, :, :])
