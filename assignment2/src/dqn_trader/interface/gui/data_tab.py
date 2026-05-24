"""DataTab — runs the data pipeline and displays resulting tensor shapes."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QFormLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dqn_trader.sdk.sdk import TradingSDK


class DataTab(QWidget):
    """Tab widget for running the data pipeline and displaying tensor shapes."""

    def __init__(self, sdk: TradingSDK) -> None:
        super().__init__()
        self._sdk = sdk
        layout = QVBoxLayout(self)
        self._ticker = QLineEdit(str(sdk.config.get("data.ticker", "AAPL")))
        form = QFormLayout()
        form.addRow("Ticker", self._ticker)
        layout.addLayout(form)
        btn = QPushButton("Prepare data")
        btn.clicked.connect(self._on_prepare)
        layout.addWidget(btn)
        self._output = QTextEdit(readOnly=True)
        layout.addWidget(self._output)

    def _on_prepare(self) -> None:
        try:
            out = self._sdk.prepare_data(self._ticker.text() or None)
        except Exception as exc:  # noqa: BLE001
            self._output.setPlainText(f"Error: {exc}")
            return
        self._output.setPlainText(
            f"train: {out.train.features.shape}\n"
            f"val  : {out.val.features.shape}\n"
            f"test : {out.test.features.shape}"
        )
