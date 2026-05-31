"""Shared base for REINFORCE + A2C tabs — both plot per-episode reward.

Each subclass overrides ``algo_name`` and ``_train_fn`` to point at the
corresponding SDK method.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fitness_rl.interface.gui.plot_widget import PlotWidget
from fitness_rl.interface.gui.worker import TrainingWorker
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.shared.types import EpisodeMetrics


class AlgoTab(QWidget):
    """One "Train" button + episodes spinbox + reward curve."""

    algo_name: str = "algo"  # overridden in subclass

    def __init__(self, sdk: FitnessRL, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self._train_btn = QPushButton(f"Train {self.algo_name}")
        self._train_btn.clicked.connect(self._on_train)
        controls.addWidget(self._train_btn)
        controls.addWidget(QLabel("Episodes:"))
        self._episodes = QSpinBox()
        self._episodes.setRange(1, 5000)
        self._episodes.setValue(50)
        controls.addWidget(self._episodes)
        self._status = QLabel("ready")
        controls.addWidget(self._status)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _train_fn(self) -> Callable[[int], list[EpisodeMetrics]]:
        """Subclasses override to return either sdk.train_reinforce or sdk.train_a2c."""
        raise NotImplementedError

    def _on_train(self) -> None:
        self._train_btn.setEnabled(False)
        self._status.setText("training…")
        n = self._episodes.value()
        train = self._train_fn()

        def fn() -> list[EpisodeMetrics]:
            self._sdk.prepare_data()
            return train(n)

        self._worker = TrainingWorker(fn, parent=self)
        self._worker.finished_with_result.connect(self._on_trained)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_trained(self, history: list[EpisodeMetrics]) -> None:
        self._train_btn.setEnabled(True)
        rewards = [m.total_reward for m in history]
        self._status.setText(f"episodes={len(rewards)} final={rewards[-1]:.4f}")

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.plot(rewards, label="total reward")  # type: ignore[attr-defined]
            ax.set_title(f"{self.algo_name} reward curve")  # type: ignore[attr-defined]
            ax.set_xlabel("Episode")  # type: ignore[attr-defined]
            ax.set_ylabel("Total reward")  # type: ignore[attr-defined]
            ax.legend()  # type: ignore[attr-defined]
        self._plot.draw(plot)

    def _on_failed(self, exc: Exception) -> None:  # pragma: no cover
        self._train_btn.setEnabled(True)
        self._status.setText(f"Failed: {exc!r}")


class ReinforceTab(AlgoTab):
    algo_name = "REINFORCE"

    def _train_fn(self) -> Callable[[int], list[EpisodeMetrics]]:
        return lambda n: self._sdk.train_reinforce(episodes=n)


class A2CTab(AlgoTab):
    algo_name = "A2C"

    def _train_fn(self) -> Callable[[int], list[EpisodeMetrics]]:
        return lambda n: self._sdk.train_a2c(episodes=n)
