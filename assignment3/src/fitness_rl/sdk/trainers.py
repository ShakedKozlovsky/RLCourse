"""Module-level trainer constructors — keeps ``sdk/sdk.py`` ≤ 150 LOC.

Each ``build_*_service`` reads its hyperparameters from the ``ConfigManager`` and
returns a fully-configured service ready to call ``.fit()`` on. The SDK methods
become 1–2 line wrappers around these.
"""

from __future__ import annotations

from fitness_rl.services.a2c_service import A2CService
from fitness_rl.services.ppo_service import PPOService
from fitness_rl.services.reinforce_service import ReinforceService
from fitness_rl.services.world_model_service import WorldModelService
from fitness_rl.shared.config import ConfigManager


def build_world_model_service(cfg: ConfigManager) -> WorldModelService:
    return WorldModelService(
        window_size=int(cfg.get("world_model.window_size")),
        lr=float(cfg.get("world_model.lr")),
        batch_size=int(cfg.get("world_model.batch_size")),
        epochs=int(cfg.get("world_model.epochs")),
        early_stop_patience=int(cfg.get("world_model.early_stop_patience")),
        train_pct=float(cfg.get("world_model.train_pct")),
    )


def build_reinforce_service(cfg: ConfigManager) -> ReinforceService:
    return ReinforceService(
        gamma=float(cfg.get("env.gamma")),
        lr=float(cfg.get("reinforce.lr")),
        use_baseline=bool(cfg.get("reinforce.use_baseline")),
        entropy_bonus=float(cfg.get("reinforce.entropy_bonus")),
        use_action_mask=bool(cfg.get("env.action_masking_enabled")),
    )


def build_a2c_service(cfg: ConfigManager) -> A2CService:
    return A2CService(
        gamma=float(cfg.get("env.gamma")),
        actor_lr=float(cfg.get("a2c.actor_lr")),
        critic_lr=float(cfg.get("a2c.critic_lr")),
        entropy_bonus=float(cfg.get("a2c.entropy_bonus")),
        use_action_mask=bool(cfg.get("env.action_masking_enabled")),
    )


def build_ppo_service(cfg: ConfigManager) -> PPOService:
    return PPOService(
        gamma=float(cfg.get("env.gamma")),
        lr=float(cfg.get("ppo.lr", cfg.get("a2c.actor_lr"))),
        clip_eps=float(cfg.get("ppo.clip_eps", 0.2)),
        n_epochs_per_batch=int(cfg.get("ppo.n_epochs_per_batch", 4)),
        n_steps_per_update=int(cfg.get("env.episode_length")),
        entropy_coef=float(cfg.get("ppo.entropy_coef",
                                     cfg.get("a2c.entropy_bonus"))),
        use_action_mask=bool(cfg.get("env.action_masking_enabled")),
    )
