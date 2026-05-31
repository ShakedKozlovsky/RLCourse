"""Compare tab — train both algos then plot reward curves + action distribution."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fitness_rl.interface.gui.plot_widget import PlotWidget
from fitness_rl.interface.gui.worker import TrainingWorker
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.comparison_service import ComparisonResult


class CompareTab(QWidget):
    """Trains both algos for N episodes and renders the side-by-side comparison."""

    def __init__(self, sdk: FitnessRL, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self._run_btn = QPushButton("Run comparison")
        self._run_btn.clicked.connect(self._on_run)
        controls.addWidget(self._run_btn)
        controls.addWidget(QLabel("Episodes per algo:"))
        self._episodes = QSpinBox()
        self._episodes.setRange(1, 5000)
        self._episodes.setValue(50)
        controls.addWidget(self._episodes)
        self._status = QLabel("ready")
        controls.addWidget(self._status)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(110)
        layout.addWidget(self._summary)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _on_run(self) -> None:
        self._run_btn.setEnabled(False)
        self._status.setText("training both…")
        n = self._episodes.value()

        def fn() -> ComparisonResult:
            self._sdk.prepare_data()
            self._sdk.train_reinforce(episodes=n)
            self._sdk.train_a2c(episodes=n)
            return self._sdk.compare()

        self._worker = TrainingWorker(fn, parent=self)
        self._worker.finished_with_result.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result: ComparisonResult) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(f"winner={result.winner}")
        self._summary.setPlainText(
            f"REINFORCE: final_mean={result.reinforce.mean_final_reward:.4f} "
            f"final_cv={result.reinforce.final_cv:.4f}\n"
            f"A2C: final_mean={result.a2c.mean_final_reward:.4f} "
            f"final_cv={result.a2c.final_cv:.4f}\n"
            f"winner={result.winner}"
        )

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.plot(result.reinforce_rewards, label="REINFORCE")  # type: ignore[attr-defined]
            ax.plot(result.a2c_rewards, label="A2C")  # type: ignore[attr-defined]
            ax.set_title("REINFORCE vs A2C — per-episode reward")  # type: ignore[attr-defined]
            ax.set_xlabel("Episode")  # type: ignore[attr-defined]
            ax.set_ylabel("Total reward")  # type: ignore[attr-defined]
            ax.legend()  # type: ignore[attr-defined]
        self._plot.draw(plot)

    def _on_failed(self, exc: Exception) -> None:  # pragma: no cover
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {exc!r}")
