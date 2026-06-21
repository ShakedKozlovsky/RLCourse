"""End-to-end DDPG training loop — slide 8 of L09.

Each iteration:
  1. observe obs ← env
  2. act    a = clip(actor(obs) + noise, -1, 1)
  3. step   (obs', r, done) ← env.step(a)
  4. push   (obs, a, r, obs', done) → replay buffer
  5. update (if buffer warm) apply_update(net, batch, ...)
  6. log    StepDiagnostic into TrainResult"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from roomba_lab.environment.roomba_env import RoombaEnv
from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.ou import OUNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.services.ddpg_update import apply_update
from roomba_lab.shared.types import EpisodeMetrics, StepDiagnostic, TrainResult


@dataclass(frozen=True)
class DDPGHyperparams:
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    batch_size: int
    warmup_steps: int
    max_grad_norm: float
    log_interval: int


class DDPGService:
    def __init__(
        self,
        net: ActorCriticNet,
        env: RoombaEnv,
        buffer: ReplayBuffer,
        noise: GaussianNoise | OUNoise,
        schedule: LinearSigmaSchedule,
        hp: DDPGHyperparams,
        device: torch.device | None = None,
    ) -> None:
        self.net = net
        self.env = env
        self.buffer = buffer
        self.noise = noise
        self.schedule = schedule
        self.hp = hp
        self.device = device or torch.device("cpu")
        self.actor_opt = torch.optim.Adam(net.actor.parameters(), lr=hp.actor_lr)
        self.critic_opt = torch.optim.Adam(net.critic.parameters(), lr=hp.critic_lr)

    def _select_action(self, obs: np.ndarray, step: int) -> np.ndarray:
        if step < self.hp.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=(self.env.action_dim,)).astype(np.float32)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.net.actor(obs_t).cpu().numpy()[0]
        action += self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def fit(self, total_timesteps: int, seed: int = 0) -> TrainResult:
        """Fit."""
        from roomba_lab.shared.types import Transition  # local: avoid cycle
        result = TrainResult()
        obs = self.env.reset(seed=seed)
        episode_reward = 0.0
        last_metrics = EpisodeMetrics(0.0, 0, 0.0, 0)
        for step in range(total_timesteps):
            self.noise.set_sigma(self.schedule.at(step))
            action = self._select_action(obs, step)
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.buffer.push(Transition(state=obs, action=action,
                                         reward=float(reward),
                                         next_state=next_obs, done=bool(done)))
            obs = next_obs
            actor_l = critic_l = mean_q = 0.0
            if len(self.buffer) >= max(self.hp.batch_size, self.hp.warmup_steps):
                batch = self.buffer.sample(self.hp.batch_size)
                diag = apply_update(self.net, batch, self.hp.gamma, self.hp.tau,
                                     self.actor_opt, self.critic_opt,
                                     max_grad_norm=self.hp.max_grad_norm,
                                     device=self.device)
                actor_l, critic_l, mean_q = diag.actor_loss, diag.critic_loss, diag.mean_q
            if step % self.hp.log_interval == 0:
                result.diagnostics.append(StepDiagnostic(
                    step=step, actor_loss=actor_l, critic_loss=critic_l,
                    mean_q=mean_q, sigma=self.noise.sigma,
                    episode_reward=episode_reward,
                    coverage=info["coverage"],
                ))
            if done:
                last_metrics = EpisodeMetrics(
                    reward=episode_reward, length=info["step"],
                    coverage=info["coverage"], collisions=info["collisions"],
                )
                obs = self.env.reset(seed=seed + step)
                episode_reward = 0.0
                self.noise.reset()
        result.final_metrics = last_metrics
        return result
