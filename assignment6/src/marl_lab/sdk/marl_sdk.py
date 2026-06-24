"""Public Python SDK — the single entry-point for library users.

Wraps services/ so importing one module is enough::

    from marl_lab.sdk.marl_sdk import MarlSDK
    sdk = MarlSDK.from_yaml('configs/setup.yaml')
    sdk.train(n_episodes=500)
    report = sdk.play_game(group_name='X', group_code='ABCDE123', ...)

The SDK is what mcp/*, cli/, gui/, and notebook all consume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.game_runner import GameRunner, RunnerConfig
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig
from marl_lab.shared.config import ConfigManager
from marl_lab.shared.logger import get_logger
from marl_lab.shared.seed import set_global_seed
from marl_lab.shared.types import GameReport, StudentEntry

LOG = get_logger("sdk")


@dataclass(frozen=True)
class SDKConfig:
    """Minimal SDK-level config (subset of full yaml)."""
    algo: str = "qmix"
    seed: int = 0
    n_episodes: int = 500


class MarlSDK:
    """High-level facade — owns trainer + game_runner. Created from yaml or args."""

    def __init__(self, cfg_path: str | Path) -> None:
        self.config = ConfigManager(setup_path=Path(cfg_path))
        seed = int(self.config.get("seed", 0))
        set_global_seed(seed)
        algo = self.config.get("marl.algorithm", "qmix")
        # Build env
        env_cfg = EnvConfig(
            grid_size=tuple(self.config.get("game.grid_size", (5, 5))),
            max_moves=int(self.config.get("game.max_moves", 25)),
            max_barriers=int(self.config.get("game.max_barriers", 5)),
            enable_barriers=bool(self.config.get("game.enable_barriers", True)),
            observation_radius=int(self.config.get("game.observation_radius", 2)),
        )
        reward_cfg = RewardConfig()
        self._rng = np.random.default_rng(seed)
        self.env = DecPomdpEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, rng=self._rng)
        self.env.reset(seed=seed)
        # Build trainer (yaml uses warmup_STEPS but trainer uses warmup_episodes;
        # we re-scale by an approximate episode length here as a coarse map)
        warmup_steps = int(self.config.get("marl.warmup_steps", 500))
        warmup_episodes = max(1, warmup_steps // 25)
        trainer_cfg = TrainerConfig(
            algo=algo,
            gamma=float(self.config.get("marl.gamma", 0.99)),
            tau=float(self.config.get("marl.tau", 0.005)),
            lr=float(self.config.get("marl.critic_lr", 1e-3)),
            batch_size=int(self.config.get("marl.batch_size", 32)),
            buffer_capacity=int(self.config.get("marl.replay_capacity", 1000)),
            warmup_episodes=warmup_episodes,
            max_seq_len=int(self.config.get("game.max_moves", 25)),
            embed_dim=int(self.config.get("marl.embed_dim", 32)),
            hyper_hidden=int(self.config.get("marl.hyper_hidden", 64)),
            gru_hidden_size=int(self.config.get("marl.rnn_hidden_size", 64)),
            hidden_sizes=tuple(self.config.get("marl.hidden_sizes", [128, 128])),
        )
        sched = LinearEpsilonSchedule(
            initial=float(self.config.get("exploration.epsilon_initial", 1.0)),
            final=float(self.config.get("exploration.epsilon_final", 0.05)),
            decay_steps=int(self.config.get("exploration.decay_steps", 1000)),
        )
        self.trainer = MarlTrainer(env=self.env, cfg=trainer_cfg,
                                     epsilon_schedule=sched, rng=self._rng)
        # Game runner (separate env to avoid state-mix with the trainer)
        self.runner = GameRunner(
            runner_cfg=RunnerConfig(
                n_sub_games=int(self.config.get("game.num_games", 6)),
                grid_size=env_cfg.grid_size,
                max_moves=env_cfg.max_moves,
                max_barriers=env_cfg.max_barriers,
                enable_barriers=env_cfg.enable_barriers,
                observation_radius=env_cfg.observation_radius,
            ),
            reward_cfg=reward_cfg,
            rng=np.random.default_rng(seed + 1),
        )

    def train(self, n_episodes: int | None = None) -> list:
        """Train for ``n_episodes`` (or yaml default). Returns per-episode history."""
        n = n_episodes if n_episodes is not None else int(self.config.get("training.total_episodes", 500))
        LOG.info("train: algo=%s episodes=%d", self.trainer.cfg.algo, n)
        history = self.trainer.train(n)
        wins = sum(1 for h in history if h.winner == "cop")
        LOG.info("train done. cop wins: %d / %d (%.1f%%)", wins, len(history),
                  100 * wins / max(1, len(history)))
        return history

    def play_game(self, group_name: str, group_code: str, github_repo: str,
                   students: list[StudentEntry],
                   timezone_name: str = "Asia/Jerusalem",
                   seed: int = 0,
                   q_b: object | None = None) -> GameReport:
        """Play a 6-sub-game adversarial round between policy_a (trainer) and
        policy_b (if provided) or a copy of policy_a (self-play default)."""
        q_a = self.trainer.q_nets["cop"]   # any of the two — both are co-trained
        # If no policy_b is provided, use the SAME network (self-play); the
        # adversarial dynamics still differ across sub-games because the env
        # seed differs.
        if q_b is None:
            q_b = self.trainer.q_nets["cop"]
        return self.runner.play_full_game(
            q_a=q_a, q_b=q_b, students=students, group_name=group_name,
            group_code=group_code, github_repo=github_repo,
            timezone_name=timezone_name, seed=seed,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save trainer Q-net + mixer weights to ``path`` (.pt)."""
        ckpt = {
            "algo": self.trainer.cfg.algo,
            "q_nets": {a: self.trainer.q_nets[a].state_dict() for a in ("cop", "thief")},
        }
        if self.trainer.mixer is not None:
            ckpt["mixer"] = self.trainer.mixer.state_dict()
        torch.save(ckpt, str(path))
        LOG.info("checkpoint saved: %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore trainer Q-net + mixer weights from ``path``."""
        ckpt = torch.load(str(path), map_location=self.trainer.device, weights_only=True)
        for a in ("cop", "thief"):
            self.trainer.q_nets[a].load_state_dict(ckpt["q_nets"][a])
        if self.trainer.mixer is not None and "mixer" in ckpt:
            self.trainer.mixer.load_state_dict(ckpt["mixer"])
        LOG.info("checkpoint loaded: %s", path)
