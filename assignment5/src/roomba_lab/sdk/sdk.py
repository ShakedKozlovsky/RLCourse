"""RoombaLab — the single facade consumed by CLI, GUI, notebook, and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from roomba_lab.environment.roomba_env import RoombaEnv
from roomba_lab.sdk.env_builder import build_env
from roomba_lab.sdk.trainers import build_ddpg_service
from roomba_lab.services.evaluation_service import EvaluationService
from roomba_lab.shared.config import ConfigManager
from roomba_lab.shared.seed import set_global_seed
from roomba_lab.shared.types import TrainResult


class RoombaLab:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config = ConfigManager(setup_path=config_path) if config_path \
            else ConfigManager()
        set_global_seed(int(self.config.get("seed")))

    def make_env(self, map_id: str | None = None,
                 max_episode_steps: int | None = None) -> RoombaEnv:
        return build_env(self.config, map_id=map_id, max_episode_steps=max_episode_steps)

    def train(self, total_timesteps: int | None = None, seed: int = 0,
              map_id: str | None = None) -> TrainResult:
        env = self.make_env(map_id=map_id)
        service = build_ddpg_service(self.config, env)
        steps = total_timesteps or int(self.config.get("training.total_timesteps"))
        return service.fit(total_timesteps=steps, seed=seed)

    def evaluate(self, net, n_episodes: int | None = None, seed: int = 0,
                  map_id: str | None = None) -> dict[str, float]:
        env = self.make_env(map_id=map_id)
        evaluator = EvaluationService(net, env)
        eps = evaluator.rollout(
            n_episodes=n_episodes or int(self.config.get("training.n_eval_episodes")),
            seed=seed,
        )
        return evaluator.aggregate(eps)

    def predict(self, net, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = net.actor(torch.as_tensor(obs).unsqueeze(0)).cpu().numpy()[0]
        return np.clip(action, -1, 1).astype(np.float32)
