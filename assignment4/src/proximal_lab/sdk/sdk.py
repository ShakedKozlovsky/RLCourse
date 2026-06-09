"""ProximalLab SDK — the single facade the CLI and GUI both call into."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.sdk.env_builder import build_vector_env
from proximal_lab.sdk.trainers import build_actor_critic, build_ppo_service
from proximal_lab.services.evaluation_service import (
    EvaluationResult,
    EvaluationService,
)
from proximal_lab.shared.config import ConfigManager
from proximal_lab.shared.logger import get_logger
from proximal_lab.shared.seed import set_global_seed
from proximal_lab.shared.types import TrainResult

_logger = get_logger(__name__)


class ProximalLab:
    """High-level facade. Holds state across calls (env, net, train result)."""

    def __init__(self, config_path: Path):
        self._cfg = ConfigManager(Path(config_path))
        set_global_seed(int(self._cfg.get("seed")))
        self._net: ActorCriticNet | None = None
        self._train_result: TrainResult | None = None

    @property
    def config(self) -> ConfigManager:
        return self._cfg

    @property
    def net(self) -> ActorCriticNet | None:
        return self._net

    @property
    def train_result(self) -> TrainResult | None:
        return self._train_result

    def train_ppo(
        self,
        env_id: str | None = None,
        total_timesteps: int | None = None,
        steps_per_rollout: int | None = None,
        seed: int = 0,
    ) -> TrainResult:
        env = build_vector_env(self._cfg, env_id=env_id, seed=seed)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self._net = build_actor_critic(self._cfg, obs_dim, action_dim)
        svc = build_ppo_service(self._cfg)
        ts = total_timesteps or int(self._cfg.get("ppo.total_timesteps"))
        spr = steps_per_rollout or int(self._cfg.get("ppo.steps_per_rollout"))
        self._train_result = svc.fit(self._net, env, total_timesteps=ts,
                                        steps_per_rollout=spr)
        ckpt = self._cfg.path("checkpoints_dir") / f"{env_id or self._cfg.get('env.id')}.pt"
        self._net.save(ckpt)
        _logger.info("ppo trained: final_reward=%.2f saved=%s",
                      self._train_result.final_mean_reward, ckpt)
        return self._train_result

    def evaluate(
        self, env_id: str | None = None, n_episodes: int = 10, deterministic: bool = True,
    ) -> EvaluationResult:
        if self._net is None:
            raise RuntimeError("no policy trained yet — call train_ppo() first")
        env_id = env_id or str(self._cfg.get("env.id"))
        return EvaluationService(deterministic=deterministic).rollout(
            self._net, env_id, n_episodes=n_episodes,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return the deterministic action for a single observation."""
        if self._net is None:
            raise RuntimeError("no policy trained yet")
        import torch
        self._net.eval()
        with torch.no_grad():
            action, _, _ = self._net.act(
                torch.from_numpy(obs).float().unsqueeze(0),
                deterministic=True,
            )
        return action.squeeze(0).numpy()

    def load_checkpoint(self, path: Path | None = None) -> ActorCriticNet:
        """Load a saved network from disk (defaults to checkpoints_dir / env_id.pt)."""
        if path is None:
            path = self._cfg.path("checkpoints_dir") / f"{self._cfg.get('env.id')}.pt"
        self._net = ActorCriticNet.load(path)
        return self._net
