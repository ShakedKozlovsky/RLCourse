"""TrainTab — kicks off training off the UI thread, plots reward, reports run dir."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from dqn_trader.interface.gui.plot_widget import PlotWidget
from dqn_trader.interface.gui.workers import TrainWorker
from dqn_trader.sdk.sdk import TradingSDK


class TrainTab(QWidget):
    def __init__(self, sdk: TradingSDK) -> None:
        super().__init__()
        self._sdk = sdk
        self._worker: TrainWorker | None = None
        layout = QVBoxLayout(self)
        self._btn = QPushButton("Train")
        self._btn.clicked.connect(self._on_train)
        layout.addWidget(self._btn)
        self._status = QLabel("idle")
        layout.addWidget(self._status)
        self.plot = PlotWidget()
        layout.addWidget(self.plot)

    def _on_train(self) -> None:
        self._btn.setEnabled(False)
        self._status.setText("training…")
        self._worker = TrainWorker(self._sdk)
        self._worker.finished_with_result.connect(self._on_done)
        self._worker.start()

    def _on_done(self, payload: object) -> None:
        self._btn.setEnabled(True)
        if isinstance(payload, Exception):
            self._status.setText(f"error: {payload}")
            return
        rewards = [m.reward for m in payload.metrics]  # type: ignore[union-attr]
        self.plot.plot_metric(rewards, "Episode reward")
        final = payload.metrics[-1]  # type: ignore[union-attr]
        self._status.setText(
            f"done — val_return={final.val_return:+.4f} — run: {payload.run_dir}"  # type: ignore[union-attr]
        )
