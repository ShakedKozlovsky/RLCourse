"""Sweep tab — pick a hyperparameter, plot the cell-by-cell final reward."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from proximal_lab.interface.gui.plot_widget import PlotWidget
from proximal_lab.sdk.sdk import ProximalLab


class SweepTab(QWidget):
    """Load existing sweep results from disk and plot them."""

    def __init__(self, sdk: ProximalLab, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Sweep:"))
        self._kind = QComboBox()
        self._kind.addItems(["lambda_multiseed", "lambda", "gamma", "clip_eps"])
        ctrl.addWidget(self._kind)
        btn = QPushButton("Plot")
        btn.clicked.connect(self._on_plot)
        ctrl.addWidget(btn)
        self._status = QLabel("ready")
        ctrl.addWidget(self._status)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _on_plot(self) -> None:
        kind = self._kind.currentText()
        root = Path(self._sdk.config.path("results_dir")).parent
        path = root / "results" / "sweeps" / f"{kind}.json"
        if not path.exists():
            self._status.setText(f"missing: {path}")
            return
        data = json.loads(path.read_text())
        values = [float(c["name"].split("=")[1]) for c in data["cells"]]
        means = [c["final_reward_mean"] for c in data["cells"]]
        cis = [c["final_reward_ci_95"] for c in data["cells"]]
        self._status.setText(f"{len(data['cells'])} cells loaded")

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.errorbar(values, means, yerr=cis, marker="o", capsize=5,  # type: ignore[attr-defined]
                         color="#4477aa", linewidth=2)
            ax.set_xlabel(kind)  # type: ignore[attr-defined]
            ax.set_ylabel("final mean reward")  # type: ignore[attr-defined]
            ax.set_title(data["label"])  # type: ignore[attr-defined]
        self._plot.draw(plot)
