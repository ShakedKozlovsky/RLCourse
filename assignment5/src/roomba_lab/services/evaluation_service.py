"""Deterministic-policy rollout evaluation."""

from __future__ import annotations

import numpy as np
import torch

from roomba_lab.environment.roomba_env import RoombaEnv
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.shared.types import EpisodeMetrics


class EvaluationService:
    def __init__(self, net: ActorCriticNet, env: RoombaEnv,
                 device: torch.device | None = None) -> None:
        self.net = net
        self.env = env
        self.device = device or torch.device("cpu")

    def rollout(self, n_episodes: int = 5, seed: int = 0) -> list[EpisodeMetrics]:
        """Run `n_episodes` deterministic-policy rollouts (no exploration noise)
        from successive seeds, return per-episode metrics."""
        episodes: list[EpisodeMetrics] = []
        for ep in range(n_episodes):
            obs = self.env.reset(seed=seed + ep)
            done = False
            episode_reward = 0.0
            info = {"coverage": 0.0, "collisions": 0, "step": 0}
            while not done:
                obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.net.actor(obs_t).cpu().numpy()[0]
                obs, r, done, info = self.env.step(np.clip(action, -1, 1).astype(np.float32))
                episode_reward += r
            episodes.append(EpisodeMetrics(
                reward=episode_reward, length=info["step"],
                coverage=info["coverage"], collisions=info["collisions"],
            ))
        return episodes

    def aggregate(self, episodes: list[EpisodeMetrics]) -> dict[str, float]:
        """Reduce a list of EpisodeMetrics → mean reward + std + mean coverage + n."""
        rewards = np.array([e.reward for e in episodes])
        covs = np.array([e.coverage for e in episodes])
        return {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "mean_coverage": float(covs.mean()),
            "n_episodes": float(len(episodes)),
        }
