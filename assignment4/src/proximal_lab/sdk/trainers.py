"""Module-level constructors so ``sdk/sdk.py`` stays close to the 150-LOC cap."""

from __future__ import annotations

from proximal_lab.model.actor_critic_network import ActorCriticNet
from proximal_lab.services.ppo_service import PPOService
from proximal_lab.shared.config import ConfigManager


def build_actor_critic(cfg: ConfigManager, obs_dim: int, action_dim: int) -> ActorCriticNet:
    hidden = tuple(int(h) for h in cfg.get("actor_critic.hidden_sizes", [64, 64]))
    return ActorCriticNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden,
        log_std_init=float(cfg.get("actor_critic.log_std_init", -0.5)),
        log_std_min=float(cfg.get("actor_critic.log_std_min", -5.0)),
        log_std_max=float(cfg.get("actor_critic.log_std_max", 2.0)),
    )


def build_ppo_service(cfg: ConfigManager) -> PPOService:
    return PPOService(
        gamma=float(cfg.get("env.gamma")),
        gae_lambda=float(cfg.get("gae.lambda")),
        lr=float(cfg.get("ppo.lr")),
        clip_eps=float(cfg.get("ppo.clip_eps")),
        n_epochs_per_update=int(cfg.get("ppo.n_epochs_per_update")),
        minibatch_size=int(cfg.get("ppo.minibatch_size")),
        value_coef=float(cfg.get("ppo.value_coef")),
        entropy_coef=float(cfg.get("ppo.entropy_coef")),
        max_grad_norm=float(cfg.get("ppo.max_grad_norm")),
        target_kl_stop=cfg.get("ppo.target_kl_stop"),
    )
