"""BacktestTab — load a checkpoint, run a backtest off-thread, plot equity."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from dqn_trader.interface.gui._checkpoint_picker import CheckpointPicker
from dqn_trader.interface.gui.plot_widget import PlotWidget
from dqn_trader.interface.gui.workers import BacktestWorker
from dqn_trader.sdk.sdk import TradingSDK


class BacktestTab(QWidget):
    """Tab widget for loading a checkpoint and running a backtest."""

    def __init__(self, sdk: TradingSDK) -> None:
        super().__init__()
        self._sdk = sdk
        layout = QVBoxLayout(self)
        self._picker = CheckpointPicker()
        layout.addWidget(self._picker)
        self._btn = QPushButton("Run backtest")
        self._btn.clicked.connect(self._on_run)
        layout.addWidget(self._btn)
        self._status = QLabel("idle")
        layout.addWidget(self._status)
        self.plot = PlotWidget()
        layout.addWidget(self.plot)
        self._worker: BacktestWorker | None = None

    def _on_run(self) -> None:
        if not self._picker.path():
            self._status.setText("pick a checkpoint first")
            return
        self._btn.setEnabled(False)
        self._status.setText("running…")
        self._worker = BacktestWorker(self._sdk, Path(self._picker.path()))
        self._worker.finished_with_result.connect(self._on_done)
        self._worker.start()

    def _on_done(self, payload: object) -> None:
        self._btn.setEnabled(True)
        if isinstance(payload, Exception):
            self._status.setText(f"error: {payload}")
            return
        self.plot.plot_equity(payload.equity, payload.benchmark)  # type: ignore[attr-defined]
        m = payload.metrics  # type: ignore[attr-defined]
        self._status.setText(
            f"return={m.total_return:+.3%}  sharpe={m.sharpe:.2f}  "
            f"max_dd={m.max_drawdown:.2%}  trades={m.n_trades}"
        )
