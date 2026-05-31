"""World-model tab — train the LSTM, plot training + validation loss curves."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fitness_rl.interface.gui.plot_widget import PlotWidget
from fitness_rl.interface.gui.worker import TrainingWorker
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.world_model_service import TrainResult


class WorldModelTab(QWidget):
    """Trains the LSTM world model + plots loss curves."""

    def __init__(self, sdk: FitnessRL, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self._train_btn = QPushButton("Train world model")
        self._train_btn.clicked.connect(self._on_train)
        controls.addWidget(self._train_btn)
        self._status = QLabel("ready")
        controls.addWidget(self._status)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _on_train(self) -> None:
        self._train_btn.setEnabled(False)
        self._status.setText("training…")

        def fn() -> TrainResult:
            self._sdk.prepare_data()
            return self._sdk.train_world_model()

        self._worker = TrainingWorker(fn, parent=self)
        self._worker.finished_with_result.connect(self._on_trained)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_trained(self, result: TrainResult) -> None:
        self._train_btn.setEnabled(True)
        self._status.setText(
            f"best_val={result.best_val_loss:.6f} epoch={result.best_epoch}"
        )

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.plot(result.train_losses, label="train")  # type: ignore[attr-defined]
            ax.plot(result.val_losses, label="val")  # type: ignore[attr-defined]
            ax.set_title("LSTM world-model loss")  # type: ignore[attr-defined]
            ax.set_xlabel("Epoch")  # type: ignore[attr-defined]
            ax.set_ylabel("MSE")  # type: ignore[attr-defined]
            ax.legend()  # type: ignore[attr-defined]
        self._plot.draw(plot)

    def _on_failed(self, exc: Exception) -> None:  # pragma: no cover
        self._train_btn.setEnabled(True)
        self._status.setText(f"Failed: {exc!r}")
