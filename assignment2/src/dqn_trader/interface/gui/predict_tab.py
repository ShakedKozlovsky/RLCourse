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
        # Feature contribution: which channels had extreme values in the last day
        last_day = market[-1]  # shape (8,) — last day of the 30-day window
        feature_names = [
            "log_return", "rsi_14", "macd", "macd_signal",
            "macd_hist", "bb_pct", "vwap_dist", "volume_norm",
        ]
        ranked = sorted(
            zip(feature_names, last_day, strict=True), key=lambda x: abs(x[1]), reverse=True
        )
        top3 = ", ".join(f"{n}={v:+.2f}" for n, v in ranked[:3])
        q_str = ", ".join(
            f"{a}: {q:.4f}" for a, q in zip(["Sell", "Hold", "Buy"], d.q_values, strict=True)
        )
        self._out.setPlainText(
            f"Recommended action: {d.action.name}\n"
            f"Confidence: {d.confidence:.1%}\n\n"
            f"Q-values: {q_str}\n\n"
            f"Explanation: top contributing features (by magnitude in last day):\n"
            f"  {top3}\n\n"
            f"The agent chose {d.action.name} because Q({d.action.name}) = "
            f"{d.q_values[d.action.value]:.4f} is the highest Q-value."
        )
