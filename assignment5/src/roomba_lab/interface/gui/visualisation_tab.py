"""GUI tab — pick a checkpoint and render an evaluation episode's trajectory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PyQt6 import QtWidgets

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.sdk.sdk import RoombaLab
from roomba_lab.tools.viz.plots import plot_trajectory_overlay

CHECKPOINT_DEFAULT = (
    Path(__file__).resolve().parents[4]
    / "saved_models" / "headline_policy.pt"
)


class VisualisationTab(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.checkpoint_path = QtWidgets.QLineEdit(str(CHECKPOINT_DEFAULT))
        self.run_button = QtWidgets.QPushButton("Render trajectory")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Checkpoint path:"))
        layout.addWidget(self.checkpoint_path)
        layout.addWidget(self.run_button)
        layout.addWidget(self.log)

    def _on_run_clicked(self) -> None:
        ckpt = Path(self.checkpoint_path.text())
        if not ckpt.exists():
            self.log.append(f"checkpoint not found: {ckpt}")
            return
        self.log.append(f"loading {ckpt} …")
        lab = RoombaLab()
        env = lab.make_env(max_episode_steps=200)
        net = ActorCriticNet(
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
            critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
        )
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        obs = env.reset(seed=0)
        for _ in range(200):
            with torch.no_grad():
                action = net.actor(torch.as_tensor(obs).unsqueeze(0)).cpu().numpy()[0]
            obs, _, done, _ = env.step(np.clip(action, -1, 1).astype(np.float32))
            if done:
                break
        loader = HouseExpoLoader(lab.config.path("data_dir") / "raw" / "sample_maps")
        polygon = loader.load(loader.map_ids()[0]).verts
        out = Path("/tmp/roomba_gui_trajectory.png")
        plot_trajectory_overlay([(p.x, p.y) for p in env.robot.trajectory],
                                  polygon, out, title="GUI eval trajectory")
        self.log.append(f"wrote {out}  coverage={env.world.coverage_fraction():.3f}")
