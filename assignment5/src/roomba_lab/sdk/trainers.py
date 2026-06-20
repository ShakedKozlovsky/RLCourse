"""Constructor layer: turns a ConfigManager + env into a ready-to-fit DDPGService."""

from __future__ import annotations

import numpy as np
import torch

from roomba_lab.environment.roomba_env import RoombaEnv
from roomba_lab.memory.replay_buffer import ReplayBuffer
from roomba_lab.model.actor_critic_network import ActorCriticNet
from roomba_lab.noise.gaussian import GaussianNoise
from roomba_lab.noise.ou import OUNoise
from roomba_lab.noise.schedule import LinearSigmaSchedule
from roomba_lab.services.ddpg_service import DDPGHyperparams, DDPGService
from roomba_lab.shared.config import ConfigManager


def build_noise(cfg: ConfigManager, action_dim: int,
                rng: np.random.Generator) -> GaussianNoise | OUNoise:
    kind = str(cfg.get("noise.kind", "gaussian"))
    sigma = float(cfg.get("noise.sigma_initial"))
    if kind == "ou":
        return OUNoise(action_dim=action_dim,
                       theta=float(cfg.get("noise.ou_theta")),
                       mu=float(cfg.get("noise.ou_mu")),
                       sigma=float(cfg.get("noise.ou_sigma")), rng=rng)
    return GaussianNoise(action_dim=action_dim, sigma=sigma, rng=rng)


def build_ddpg_service(cfg: ConfigManager, env: RoombaEnv,
                        rng: np.random.Generator | None = None) -> DDPGService:
    rng = rng or np.random.default_rng(int(cfg.get("seed")))
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(cfg.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(cfg.get("ddpg.critic_hidden_sizes")),
    )
    buffer = ReplayBuffer(capacity=int(cfg.get("ddpg.replay_capacity")),
                           obs_dim=env.obs_dim, action_dim=env.action_dim, rng=rng)
    noise = build_noise(cfg, env.action_dim, rng)
    schedule = LinearSigmaSchedule(
        initial=float(cfg.get("noise.sigma_initial")),
        final=float(cfg.get("noise.sigma_final")),
        decay_steps=int(cfg.get("noise.decay_steps")),
    )
    hp = DDPGHyperparams(
        gamma=float(cfg.get("ddpg.gamma")),
        tau=float(cfg.get("ddpg.tau")),
        actor_lr=float(cfg.get("ddpg.actor_lr")),
        critic_lr=float(cfg.get("ddpg.critic_lr")),
        batch_size=int(cfg.get("ddpg.batch_size")),
        warmup_steps=int(cfg.get("ddpg.warmup_steps")),
        max_grad_norm=float(cfg.get("ddpg.max_grad_norm")),
        log_interval=int(cfg.get("training.log_interval")),
    )
    device = torch.device(str(cfg.get("device", "cpu")))
    return DDPGService(net, env, buffer, noise, schedule, hp, device=device)
