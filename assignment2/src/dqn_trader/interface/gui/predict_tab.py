"""PredictTab — show the agent's decision on the most recent test window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QPushButton, QTextEdit, QVBoxLayout, QWidget

from dqn_trader.interface.gui._checkpoint_picker import CheckpointPicker
from dqn_trader.sdk.sdk import TradingSDK


class PredictTab(QWidget):
    """Tab widget for single-step action prediction from the latest window."""

    def __init__(self, sdk: TradingSDK) -> None:
        super().__init__()
        self._sdk = sdk
        layout = QVBoxLayout(self)
        self._picker = CheckpointPicker()
        layout.addWidget(self._picker)
        btn = QPushButton("Predict next action")
        btn.clicked.connect(self._on_predict)
        layout.addWidget(btn)
        self._out = QTextEdit(readOnly=True)
        layout.addWidget(self._out)

    def _on_predict(self) -> None:
        try:
            pipeline = self._sdk.prepare_data()
            market = pipeline.test.features[-1]
            d = self._sdk.predict(market, checkpoint=Path(self._picker.path()))
        except Exception as exc:  # noqa: BLE001
            self._out.setPlainText(f"error: {exc}")
            return
        self._out.setPlainText(
            f"action: {d.action.name}\nconfidence: {d.confidence:.3f}\n"
            f"Q-values: {d.q_values.tolist()}"
        )
