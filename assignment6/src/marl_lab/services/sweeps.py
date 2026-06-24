"""Sweep runner — execute a Cartesian product of yaml overrides.

PRD § 6 plans 4 empirical studies; this is the engine. Each cell is one
(algo, grid, radius, seed) tuple; we evaluate the trained policy on a fixed
eval set and record (cop_win_rate, mean_moves, final_critic_loss)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig
from marl_lab.shared.logger import get_logger

LOG = get_logger("sweeps")


@dataclass
class SweepCellSpec:
    """One sweep cell — overrides applied on top of a baseline TrainerConfig."""
    algo: str
    grid_size: tuple[int, int]
    observation_radius: int
    seed: int
    n_episodes: int = 200       # tiny by default; full studies override


@dataclass
class SweepCellResult:
    """Per-cell metrics — mirrors KPIs from PRD § 7."""
    spec: SweepCellSpec
    cop_win_rate: float
    mean_moves: float
    mean_critic_loss: float
    n_episodes: int


@dataclass
class SweepResults:
    """Aggregated sweep output — list of cell results."""
    cells: list[SweepCellResult] = field(default_factory=list)

    def to_table(self) -> list[dict[str, Any]]:
        """Flatten to a list of dicts (notebook-friendly)."""
        out: list[dict[str, Any]] = []
        for r in self.cells:
            out.append({
                "algo": r.spec.algo,
                "grid_size": str(r.spec.grid_size),
                "observation_radius": r.spec.observation_radius,
                "seed": r.spec.seed,
                "cop_win_rate": r.cop_win_rate,
                "mean_moves": r.mean_moves,
                "mean_critic_loss": r.mean_critic_loss,
                "n_episodes": r.n_episodes,
            })
        return out


def run_one_cell(spec: SweepCellSpec) -> SweepCellResult:
    """Train one trainer for ``n_episodes`` and return the per-cell metrics."""
    env = DecPomdpEnv(
        env_cfg=EnvConfig(
            grid_size=spec.grid_size,
            max_moves=max(8, spec.grid_size[0] * spec.grid_size[1]),
            max_barriers=3, enable_barriers=False,
            observation_radius=spec.observation_radius,
        ),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(spec.seed),
    )
    env.reset(seed=spec.seed)
    cfg = TrainerConfig(
        algo=spec.algo, batch_size=4, buffer_capacity=64,
        warmup_episodes=2, max_seq_len=env.env_cfg.max_moves,
        embed_dim=8, hyper_hidden=16, gru_hidden_size=8,
        hidden_sizes=(16,),
    )
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=spec.n_episodes)
    trainer = MarlTrainer(env=env, cfg=cfg, epsilon_schedule=sched,
                            rng=np.random.default_rng(spec.seed))
    history = trainer.train(n_episodes=spec.n_episodes)
    wins = sum(1 for h in history if h.winner == "cop")
    cop_win_rate = wins / max(1, len(history))
    mean_moves = float(np.mean([h.episode_steps for h in history]))
    losses = [h.critic_loss for h in history if h.critic_loss != 0.0]
    mean_critic_loss = float(np.mean(losses)) if losses else 0.0
    LOG.info("sweep cell %s/%s/r=%d/seed=%d → cop_win=%.2f, moves=%.1f",
              spec.algo, spec.grid_size, spec.observation_radius, spec.seed,
              cop_win_rate, mean_moves)
    return SweepCellResult(
        spec=spec, cop_win_rate=cop_win_rate, mean_moves=mean_moves,
        mean_critic_loss=mean_critic_loss, n_episodes=len(history),
    )


def run_sweep(
    algorithms: list[str],
    grid_sizes: list[tuple[int, int]],
    observation_radii: list[int],
    seeds: list[int],
    n_episodes: int = 50,
) -> SweepResults:
    """Run the Cartesian product of (algorithm × grid × radius × seed)."""
    results = SweepResults()
    for algo, grid, r, seed in itertools.product(algorithms, grid_sizes,
                                                    observation_radii, seeds):
        spec = SweepCellSpec(
            algo=algo, grid_size=grid, observation_radius=r, seed=seed,
            n_episodes=n_episodes,
        )
        results.cells.append(run_one_cell(spec))
    return results
