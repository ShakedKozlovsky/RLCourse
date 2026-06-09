"""Greedy / stochastic evaluation rollouts for a trained ``ActorCriticNet``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from proximal_lab.environment.mujoco_env import make_env
from proximal_lab.model.actor_critic_network import ActorCriticNet


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of an n-episode evaluation rollout."""

    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    per_episode_rewards: list[float]


class EvaluationService:
    """Run ``n_episodes`` evaluation rollouts and aggregate the returns."""

    def __init__(self, deterministic: bool = True):
        self._deterministic = bool(deterministic)

    def rollout(
        self,
        net: ActorCriticNet,
        env_id: str,
        n_episodes: int = 10,
        seed: int = 0,
        max_steps: int = 1000,
    ) -> EvaluationResult:
        if n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        env = make_env(env_id, seed=seed, training=False)
        rewards: list[float] = []
        lengths: list[int] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward, length, terminated = 0.0, 0, False
            while not terminated and length < max_steps:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                    action, _, _ = net.act(obs_t, deterministic=self._deterministic)
                action_np = action.squeeze(0).numpy()
                obs, r, term, trunc, _ = env.step(action_np)
                ep_reward += float(r)
                length += 1
                terminated = bool(term or trunc)
            rewards.append(ep_reward)
            lengths.append(length)
        env.close()
        return EvaluationResult(
            n_episodes=n_episodes,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            mean_length=float(np.mean(lengths)),
            per_episode_rewards=rewards,
        )
