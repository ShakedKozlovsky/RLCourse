"""Compare tab — overlay cross-env reward curves."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from proximal_lab.interface.gui.plot_widget import PlotWidget
from proximal_lab.sdk.sdk import ProximalLab


class CompareTab(QWidget):
    """Load Layer-11 cross-env JSON and overlay the curves."""

    def __init__(self, sdk: ProximalLab, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        btn = QPushButton("Plot cross-env transfer")
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
        root = Path(self._sdk.config.path("results_dir")).parent
        path = root / "results" / "layer11" / "cross_env.json"
        if not path.exists():
            self._status.setText(f"missing: {path}")
            return
        data = json.loads(path.read_text())
        self._status.setText(f"{len(data['runs'])} envs loaded")

        def plot(ax: object) -> None:  # noqa: ANN401
            for run in data["runs"]:
                ax.plot(run["per_iteration_reward"], "-o",  # type: ignore[attr-defined]
                         label=run["env_id"], linewidth=2)
            ax.set_xlabel("iteration")  # type: ignore[attr-defined]
            ax.set_ylabel("mean episode reward")  # type: ignore[attr-defined]
            ax.set_title("Cross-env transfer")  # type: ignore[attr-defined]
            ax.legend()  # type: ignore[attr-defined]
        self._plot.draw(plot)
