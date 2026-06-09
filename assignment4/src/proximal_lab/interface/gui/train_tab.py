"""Train tab — single-config PPO training with live reward + KL + clip-fraction plot."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from proximal_lab.interface.gui.plot_widget import PlotWidget
from proximal_lab.interface.gui.worker import TrainingWorker
from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.shared.types import TrainResult


class TrainTab(QWidget):
    """Choose env + timesteps, train PPO, plot per-iteration rewards."""

    def __init__(self, sdk: ProximalLab, parent: QWidget | None = None):
        super().__init__(parent)
        self._sdk = sdk
        self._worker: TrainingWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Env:"))
        self._env = QComboBox()
        self._env.addItems(["HalfCheetah-v5", "Walker2d-v5"])
        ctrl.addWidget(self._env)
        ctrl.addWidget(QLabel("Timesteps:"))
        self._timesteps = QSpinBox()
        self._timesteps.setRange(512, 1_000_000)
        self._timesteps.setSingleStep(1024)
        self._timesteps.setValue(8192)
        ctrl.addWidget(self._timesteps)
        self._btn = QPushButton("Train")
        self._btn.clicked.connect(self._on_train)
        ctrl.addWidget(self._btn)
        self._status = QLabel("ready")
        ctrl.addWidget(self._status)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)
        self._plot = PlotWidget(self)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def _on_train(self) -> None:
        self._btn.setEnabled(False)
        self._status.setText("training…")
        env_id = self._env.currentText()
        total_ts = self._timesteps.value()

        def fn() -> TrainResult:
            return self._sdk.train_ppo(env_id=env_id, total_timesteps=total_ts,
                                         steps_per_rollout=1024)

        self._worker = TrainingWorker(fn, parent=self)
        self._worker.finished_with_result.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result: TrainResult) -> None:
        self._btn.setEnabled(True)
        self._status.setText(
            f"iters={len(result.diagnostics)} final_reward={result.final_mean_reward:.2f}"
        )
        rewards = [d.mean_episode_reward for d in result.diagnostics]
        clip_fracs = [d.clip_fraction for d in result.diagnostics]

        def plot(ax: object) -> None:  # noqa: ANN401
            ax.plot(rewards, "-o", color="#4477aa", label="episode reward")  # type: ignore[attr-defined]
            ax.set_xlabel("iteration")  # type: ignore[attr-defined]
            ax.set_ylabel("reward")  # type: ignore[attr-defined]
            ax.set_title(f"PPO training on {self._env.currentText()}")  # type: ignore[attr-defined]
            ax2 = ax.twinx()  # type: ignore[attr-defined]
            ax2.plot(clip_fracs, "-s", color="#cc6677", alpha=0.6,
                      label="clip fraction")
            ax2.set_ylabel("clip fraction", color="#cc6677")
            ax.legend(loc="upper left")  # type: ignore[attr-defined]
        self._plot.draw(plot)

    def _on_failed(self, exc: Exception) -> None:  # pragma: no cover
        self._btn.setEnabled(True)
        self._status.setText(f"failed: {exc!r}")
