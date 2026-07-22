"""Reflection Q3 empirical study: swarm-vs-single-agent pursuit.

Spec § 7 Q3 asks: "How does the swarm-vs-single-agent framing change the
optimal-policy story?"

We answer this empirically. Random-policy rollouts on the 5×5 grid with
N ∈ {1, 2, 3} cops chasing 1 thief; report cop-team capture rate as a
function of N. The trend illuminates the coordination story:
  - N = 1: baseline (spec-conforming task)
  - N = 2: two cops can 'corner' the thief; capture rate should ~2×
  - N = 3: 5×5 has only 25 cells; 3 cops occupy 12% of the grid, capture
    rate approaches 1 as coordination-through-density kicks in.

This isn't a full RL study (would need to train N-cop policies with a
proper multi-agent algorithm) but it establishes the emergent coordination
gradient. Full RL training left as future work; the aim here is empirical
grounding for the reflection answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marl_lab.environment.multi_cop_env import MultiCopEnv, MultiCopEnvConfig


def _random_rollout(n_cops: int, n_games: int, seed: int = 0) -> dict:
    """Play n_games with uniformly-random policies. Report cop-team win rate."""
    env = MultiCopEnv(
        cfg=MultiCopEnvConfig(n_cops=n_cops, grid_size=(5, 5), max_moves=25,
                                observation_radius=2),
        rng=np.random.default_rng(seed),
    )
    wins_cops = 0
    moves_list: list[int] = []
    for g in range(n_games):
        env.reset(seed=seed + g)
        rng = np.random.default_rng(seed + g + 100000)
        for _ in range(25):
            actions = {"thief": int(rng.integers(0, 5))}
            for i in range(n_cops):
                actions[f"cop_{i}"] = int(rng.integers(0, 5))
            _, _, done, info = env.step(actions)
            if done:
                if info["winner"] == "cops":
                    wins_cops += 1
                moves_list.append(info["step"])
                break
    return {
        "n_cops": n_cops,
        "n_games": n_games,
        "cop_team_win_rate": wins_cops / n_games,
        "mean_moves_per_game": float(np.mean(moves_list)) if moves_list else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=500)
    parser.add_argument("--output-dir", default="assets")
    args = parser.parse_args()

    results = []
    for n_cops in (1, 2, 3, 4):
        r = _random_rollout(n_cops=n_cops, n_games=args.n_games)
        results.append(r)
        print(f"[q3] N={r['n_cops']} cops → win rate {r['cop_team_win_rate']:.1%}, "
                f"mean {r['mean_moves_per_game']:.1f} moves")

    out_dir = Path(args.output_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "q3_swarm_vs_single.json").write_text(
        json.dumps(results, indent=2))

    # Plot the coordination gradient
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["n_cops"] for r in results]
    ys = [r["cop_team_win_rate"] for r in results]
    ax.plot(xs, ys, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Number of cops (N)")
    ax.set_ylabel("Cop-team capture rate (random policy, 500 games)")
    ax.set_title("Reflection Q3 — swarm vs single-agent pursuit\n"
                   "(random policies on 5×5 grid; 25-move cap)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "q3_swarm_vs_single.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[q3] saved {out_dir / 'figures' / 'q3_swarm_vs_single.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
