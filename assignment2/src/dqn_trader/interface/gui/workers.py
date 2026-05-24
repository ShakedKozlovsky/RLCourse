"""QThread workers for long-running SDK operations.

Why threading: PyQt6 freezes if the event loop is blocked. Training takes
many seconds even on the smallest config, so it must run off the UI thread.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from dqn_trader.sdk.sdk import TradingSDK, TrainResult


class TrainWorker(QThread):
    """Runs ``sdk.train()`` off the UI thread and emits the result when done."""

    finished_with_result = pyqtSignal(object)  # TrainResult | Exception

    def __init__(self, sdk: TradingSDK, ticker: str | None = None) -> None:
        super().__init__()
        self._sdk = sdk
        self._ticker = ticker

    def run(self) -> None:  # pragma: no cover — exercised by GUI smoke test
        """Execute sdk.train() off the UI thread and emit the result or exception."""
        try:
            result = self._sdk.train(ticker=self._ticker)
        except Exception as exc:  # noqa: BLE001
            self.finished_with_result.emit(exc)
            return
        self.finished_with_result.emit(result)


class BacktestWorker(QThread):
    """Runs ``sdk.backtest(checkpoint)`` off the UI thread."""

    finished_with_result = pyqtSignal(object)  # BacktestResult | Exception

    def __init__(self, sdk: TradingSDK, checkpoint: Path) -> None:
        super().__init__()
        self._sdk = sdk
        self._checkpoint = checkpoint

    def run(self) -> None:  # pragma: no cover
        """Execute sdk.backtest() off the UI thread and emit the result or exception."""
        try:
            result = self._sdk.backtest(self._checkpoint)
        except Exception as exc:  # noqa: BLE001
            self.finished_with_result.emit(exc)
            return
        self.finished_with_result.emit(result)


# Re-export for type hints
__all__ = ["TrainResult", "TrainWorker", "BacktestWorker"]
