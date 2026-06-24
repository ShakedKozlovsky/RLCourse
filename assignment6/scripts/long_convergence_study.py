"""Long convergence study — 500-episode runs across QMIX/QPLEX/IQL.

The default scripts/generate_artifacts.py uses 60 episodes per algo to keep
the audit fast, but 60 episodes is mostly noise. This study runs 500
episodes per algorithm on a 4×4 grid (computationally feasible in a few
minutes locally) and emits two artifacts:

  - assets/figures/long_convergence.png — smoothed cop reward curves
                                          (rolling-window 25 episodes)
  - assets/logs/long_convergence.csv    — raw per-episode metrics

Useful for substantiating the § 7.2 claim that CTDE methods converge while
IQL drifts due to non-stationarity. Run separately from the audit:
    uv run python scripts/long_convergence_study.py"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.noise.schedule import LinearEpsilonSchedule
from marl_lab.services.marl_trainer import MarlTrainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "assets" / "figures" / "long_convergence.png"
CSV_OUT = ROOT / "assets" / "logs" / "long_convergence.csv"


def _train(algo: str, n_episodes: int, seed: int) -> list:
    """Train one trainer for n_episodes; return per-episode diagnostics."""
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(4, 4), max_moves=15, max_barriers=3,
                          enable_barriers=False, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(seed),
    )
    env.reset(seed=seed)
    cfg = TrainerConfig(
        algo=algo, batch_size=16, buffer_capacity=512,
        warmup_episodes=10, max_seq_len=15,
        embed_dim=32, hyper_hidden=64,
        gru_hidden_size=32, hidden_sizes=(64, 64),
    )
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05, decay_steps=n_episodes)
    trainer = MarlTrainer(env, cfg, sched, rng=np.random.default_rng(seed))
    return trainer.train(n_episodes=n_episodes)


def _smooth(xs: list[float], window: int) -> list[float]:
    """Centred rolling mean — same length as input via boundary repeats."""
    if not xs:
        return []
    a = np.array(xs, dtype=float)
    out = np.empty_like(a)
    half = window // 2
    for i in range(len(a)):
        lo = max(0, i - half)
        hi = min(len(a), i + half + 1)
        out[i] = a[lo:hi].mean()
    return out.tolist()


def main(n_episodes: int = 500, seed: int = 0) -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, color in (("qmix", "#1f77b4"),
                         ("qplex", "#2ca02c"),
                         ("iql", "#d62728")):
        print(f"[long] training {algo} for {n_episodes} episodes…")
        history = _train(algo=algo, n_episodes=n_episodes, seed=seed)
        rewards_cop = [h.episode_reward_cop for h in history]
        smoothed = _smooth(rewards_cop, window=25)
        ax.plot(smoothed, label=f"{algo.upper()} (rolling-25)",
                color=color, linewidth=2)
        # Light raw signal in the background
        ax.plot(rewards_cop, alpha=0.18, color=color, linewidth=0.7)
        for ep_idx, raw in enumerate(rewards_cop):
            rows.append({
                "algo": algo, "episode": ep_idx,
                "raw_cop_reward": raw, "smoothed_cop_reward": smoothed[ep_idx],
                "winner": history[ep_idx].winner or "",
                "epsilon": history[ep_idx].epsilon,
                "critic_loss": history[ep_idx].critic_loss,
            })
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cop episode reward (rolling-25 mean)")
    ax.set_title(f"500-episode convergence — QMIX vs QPLEX vs IQL "
                  f"(4×4 grid, seed={seed})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    with CSV_OUT.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"saved {FIG_OUT}")
    print(f"saved {CSV_OUT}")
    return 0


if __name__ == "__main__":
    import sys
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    raise SystemExit(main(n_episodes=eps))
