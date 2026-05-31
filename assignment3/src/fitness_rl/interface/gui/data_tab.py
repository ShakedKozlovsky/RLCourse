"""Data tab — load Kaggle CSVs, show trajectory summary."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fitness_rl.interface.gui.plot_widget import PlotWidget
from fitness_rl.interface.gui.worker import TrainingWorker
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.data_service import PipelineOutput


class DataTab(QWidget):
    """Loads + summarises the synthetic trajectory."""

    def __init__(self, sdk: FitnessRL, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self._load_btn = QPushButton("Load data")
        self._load_btn.clicked.connect(self._on_load)
        controls.addWidget(self._load_btn)
        controls.addWidget(QLabel("Loads CSVs → trajectory → 16-dim states"))
        controls.addStretch(1)
        layout.addLayout(controls)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(110)
        layout.addWidget(self._summary)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _on_load(self) -> None:
        self._load_btn.setEnabled(False)
        self._worker = TrainingWorker(self._sdk.prepare_data, parent=self)
        self._worker.finished_with_result.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_loaded(self, out: PipelineOutput) -> None:
        self._load_btn.setEnabled(True)
        self._summary.setPlainText(
            f"chosen={out.chosen_title}\n"
            f"weeks={out.n_weeks}  days={len(out.trajectory)}  "
            f"state_dim={out.states.shape[1]}\n"
            f"actions distribution: " + ", ".join(
                f"{i}={int((out.actions == i).sum())}" for i in range(5)
            )
        )

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.plot(out.states[:, 0], label="volume_normalised")  # type: ignore[attr-defined]
            ax.set_title("Per-day volume (normalised)")  # type: ignore[attr-defined]
            ax.set_xlabel("Day")  # type: ignore[attr-defined]
            ax.legend()  # type: ignore[attr-defined]
        self._plot.draw(plot)

    def _on_failed(self, exc: Exception) -> None:  # pragma: no cover - UI path
        self._load_btn.setEnabled(True)
        self._summary.setPlainText(f"Failed: {exc!r}")
