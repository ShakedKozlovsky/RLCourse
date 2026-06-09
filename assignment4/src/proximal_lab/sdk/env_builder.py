"""Build a vectorised env from config."""

from __future__ import annotations

from proximal_lab.environment.vector_env import SyncVectorEnv
from proximal_lab.shared.config import ConfigManager


def build_vector_env(cfg: ConfigManager, env_id: str | None = None,
                       seed: int = 0) -> SyncVectorEnv:
    """Create ``SyncVectorEnv`` sized per config."""
    env_id = env_id or str(cfg.get("env.id"))
    n_envs = int(cfg.get("env.n_parallel_envs", 4))
    return SyncVectorEnv(env_id=env_id, n_envs=n_envs, seed=seed)
