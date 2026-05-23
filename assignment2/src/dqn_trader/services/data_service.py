"""Orchestrates the data pipeline end-to-end. Used by the SDK and training services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dqn_trader.data.feature_engineer import FEATURE_COLUMNS, FeatureEngineer
from dqn_trader.data.scaler import ZScoreScaler
from dqn_trader.data.splitter import ChronologicalSplitter, Split
from dqn_trader.data.window_builder import WindowBuilder
from dqn_trader.data.yfinance_client import YFinanceClient
from dqn_trader.shared.config import ConfigManager
from dqn_trader.shared.gatekeeper import ApiGatekeeper
from dqn_trader.shared.logger import get_logger
from dqn_trader.shared.types import SliceData

_logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineOutput:
    """End-to-end pipeline result."""

    train: SliceData
    val: SliceData
    test: SliceData
    feature_columns: tuple[str, ...]
    scaler: ZScoreScaler


class DataService:
    """High-level facade. Reads config, runs every step, returns ready-to-train tensors."""

    def __init__(self, config: ConfigManager, client: YFinanceClient | None = None):
        self._cfg = config
        if client is None:
            gate = ApiGatekeeper(self._cfg.rate_limits["services"]["yfinance"])
            cache_dir = self._cfg.path("data_raw_dir")
            client = YFinanceClient(cache_dir=cache_dir, gatekeeper=gate)
        self._client = client

    def run(self, ticker: str | None = None) -> PipelineOutput:
        ticker = ticker or str(self._cfg.get("data.ticker"))
        start = str(self._cfg.get("data.start"))
        end = str(self._cfg.get("data.end"))
        interval = str(self._cfg.get("data.interval", "1d"))
        window = int(self._cfg.get("data.window_size"))
        train_pct = float(self._cfg.get("data.train_pct"))
        val_pct = float(self._cfg.get("data.val_pct"))
        _logger.info("data pipeline: ticker=%s %s..%s", ticker, start, end)
        raw = self._client.fetch(ticker, start, end, interval)
        # Features first (all indicators are causal — no leakage), then split.
        feats = FeatureEngineer().transform(raw)
        raw_aligned = raw.reindex(feats.index)
        feat_split: Split = ChronologicalSplitter(train_pct, val_pct).split(feats)
        raw_split: Split = ChronologicalSplitter(train_pct, val_pct).split(raw_aligned)
        scaler = ZScoreScaler().fit(feat_split.train)
        builder = WindowBuilder(window)
        return PipelineOutput(
            train=_pack(
                "train", builder.build(scaler.transform(feat_split.train)), raw_split.train
            ),
            val=_pack("val", builder.build(scaler.transform(feat_split.val)), raw_split.val),
            test=_pack("test", builder.build(scaler.transform(feat_split.test)), raw_split.test),
            feature_columns=FEATURE_COLUMNS,
            scaler=scaler,
        )

    def save_scaler(self, scaler: ZScoreScaler, path: Path) -> None:
        scaler.save(path)


def _pack(name: str, features: np.ndarray, raw_slice: pd.DataFrame) -> SliceData:
    """Pack a windowed slice together with its Close series and date index."""
    return SliceData(
        name=name,
        features=features,
        raw_close=raw_slice["Close"].to_numpy(dtype=np.float32),
        dates=raw_slice.index.to_numpy(),
    )
