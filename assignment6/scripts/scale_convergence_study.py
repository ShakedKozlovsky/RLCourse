"""Scale-vs-CTDE-advantage study: does the gap between CTDE and IQL grow
with grid size? Lin et al. (2025, bib ref [12]) claims yes — this script
provides the empirical evidence on this codebase.

Trains QMIX / QPLEX / IQL on three grid sizes (5×5, 6×6, 7×7) for
``n_episodes`` per cell. Emits two artifacts:

  - assets/figures/scale_convergence.png — 3-subplot per-grid learning
    curves (rolling-25 smoother)
  - assets/figures/ctde_advantage_vs_grid.png — single-panel summary:
    final-window mean cop reward vs grid size, three lines (one per algo).
    The key visual claim: the IQL gap widens as the state space grows.
  - assets/logs/scale_convergence.csv — raw per-episode metrics

Configurable via CLI argument; defaults are sized for ~30 min of CPU."""

from __future__ import annotations

import csv
import sys
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
FIG_CONV = ROOT / "assets" / "figures" / "scale_convergence.png"
FIG_SUMMARY = ROOT / "assets" / "figures" / "ctde_advantage_vs_grid.png"
CSV_OUT = ROOT / "assets" / "logs" / "scale_convergence.csv"

GRIDS = [(5, 5), (6, 6), (7, 7)]
ALGOS = ("qmix", "qplex", "iql")
COLOURS = {"qmix": "#1f77b4", "qplex": "#2ca02c", "iql": "#d62728"}


def _train(algo: str, grid: tuple[int, int], n_episodes: int,
            seed: int = 0) -> list:
    """Run one training and return per-episode diagnostics."""
    h, _ = grid
    max_moves = max(h * h, 25)         # bigger grid → more time per sub-game
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=grid, max_moves=max_moves,
                          max_barriers=3, enable_barriers=False,
                          observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(seed),
    )
    env.reset(seed=seed)
    cfg = TrainerConfig(
        algo=algo, batch_size=16, buffer_capacity=512,
        warmup_episodes=10, max_seq_len=max_moves,
        embed_dim=32, hyper_hidden=64,
        gru_hidden_size=32, hidden_sizes=(64, 64),
    )
    sched = LinearEpsilonSchedule(initial=1.0, final=0.05,
                                    decay_steps=n_episodes)
    trainer = MarlTrainer(env, cfg, sched, rng=np.random.default_rng(seed))
    return trainer.train(n_episodes=n_episodes)


def _smooth(xs: list[float], window: int) -> list[float]:
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


def main(n_episodes: int = 250) -> int:
    FIG_CONV.parent.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[scale] running {len(GRIDS)} grids × {len(ALGOS)} algos × "
            f"{n_episodes} episodes = {len(GRIDS) * len(ALGOS) * n_episodes} total episodes")
    results: dict[tuple[str, tuple[int, int]], dict] = {}
    rows: list[dict] = []
    for grid in GRIDS:
        for algo in ALGOS:
            print(f"[scale] training {algo} on {grid}…", flush=True)
            history = _train(algo, grid, n_episodes)
            rewards = [h.episode_reward_cop for h in history]
            results[(algo, grid)] = {
                "rewards": rewards,
                "smoothed": _smooth(rewards, window=25),
                "final_mean": float(np.mean(rewards[-50:]) if len(rewards) >= 50
                                      else np.mean(rewards)),
                "winners": [h.winner or "" for h in history],
            }
            for ep_idx, r in enumerate(rewards):
                rows.append({
                    "algo": algo, "grid": f"{grid[0]}x{grid[1]}",
                    "episode": ep_idx, "cop_reward": r,
                    "winner": history[ep_idx].winner or "",
                    "epsilon": history[ep_idx].epsilon,
                    "critic_loss": history[ep_idx].critic_loss,
                })

    # ----- Figure 1: per-grid convergence (3 subplots) -----
    fig, axes = plt.subplots(1, len(GRIDS), figsize=(15, 4.5), sharey=True)
    for ax, grid in zip(axes, GRIDS, strict=True):
        for algo in ALGOS:
            res = results[(algo, grid)]
            ax.plot(res["smoothed"], color=COLOURS[algo], linewidth=2,
                     label=f"{algo.upper()}")
            ax.plot(res["rewards"], color=COLOURS[algo], alpha=0.15, linewidth=0.6)
        ax.set_title(f"{grid[0]}×{grid[1]} grid")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cop episode reward (rolling-25)")
    axes[-1].legend(loc="lower right")
    fig.suptitle(f"Scale convergence study — {n_episodes} episodes per cell",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_CONV, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_CONV}")

    # ----- Figure 2: CTDE advantage gap vs grid size -----
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    grid_labels = [f"{g[0]}×{g[1]}" for g in GRIDS]
    for algo in ALGOS:
        finals = [results[(algo, g)]["final_mean"] for g in GRIDS]
        ax2.plot(grid_labels, finals, marker="o", linewidth=2,
                   color=COLOURS[algo], label=f"{algo.upper()}")
    ax2.set_xlabel("Grid size")
    ax2.set_ylabel("Final-50 mean cop reward")
    ax2.set_title("CTDE advantage vs grid size — Lin 2025 hypothesis")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_SUMMARY, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {FIG_SUMMARY}")

    with CSV_OUT.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"saved {CSV_OUT}")

    # Console summary table
    print("\n=== FINAL-50 MEAN COP REWARD ===")
    print(f"{'algo':<8}", *[f"{f'{g[0]}x{g[1]}':>10}" for g in GRIDS])
    for algo in ALGOS:
        cells = [f"{results[(algo, g)]['final_mean']:>+10.3f}" for g in GRIDS]
        print(f"{algo:<8}", *cells)
    return 0


if __name__ == "__main__":
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    raise SystemExit(main(n_episodes=eps))
