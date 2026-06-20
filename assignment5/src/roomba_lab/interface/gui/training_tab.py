"""GUI tab — quick smoke-train + live reward/loss plot."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PyQt6 import QtWidgets  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402

from roomba_lab.sdk.sdk import RoombaLab  # noqa: E402
from roomba_lab.sdk.trainers import build_ddpg_service  # noqa: E402


class TrainingTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.steps_box = QtWidgets.QSpinBox()
        self.steps_box.setMinimum(100)
        self.steps_box.setMaximum(20000)
        self.steps_box.setValue(500)
        self.train_button = QtWidgets.QPushButton("Start short training run")
        self.train_button.clicked.connect(self._on_train_clicked)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(QtWidgets.QLabel("Training steps:"))
        layout.addWidget(self.steps_box)
        layout.addWidget(self.train_button)
        layout.addWidget(self.log)

    def _on_train_clicked(self) -> None:
        steps = int(self.steps_box.value())
        self.log.append(f"training {steps} steps …")
        lab = RoombaLab()
        env = lab.make_env()
        svc = build_ddpg_service(lab.config, env)
        result = svc.fit(total_timesteps=steps, seed=0)
        last = result.diagnostics[-1]
        self.log.append(
            f"done — coverage={last.coverage:.3f}  reward={last.episode_reward:.1f}  "
            f"critic_loss={last.critic_loss:.3f}"
        )
        self._render_curve(result)

    def _render_curve(self, result) -> None:
        fig, ax = plt.subplots(figsize=(5, 3))
        steps = [d.step for d in result.diagnostics]
        ax.plot(steps, [d.episode_reward for d in result.diagnostics],
                 label="episode reward")
        ax.plot(steps, [d.critic_loss for d in result.diagnostics],
                 label="critic loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = "/tmp/roomba_gui_curve.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        self.log.append(f"wrote {out}")
