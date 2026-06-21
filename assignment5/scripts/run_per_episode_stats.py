"""Per-episode reward distribution evaluation — Mod1 fix.

Take the 50k-step tuned policy, run N evaluation episodes from different seeds,
record the per-episode reward + coverage distributions, and emit a boxplot.
Replaces 'final reward of last episode' with proper distributional reporting."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from roomba_lab.model.actor_critic_network import ActorCriticNet  # noqa: E402
from roomba_lab.sdk.sdk import RoombaLab  # noqa: E402
from roomba_lab.services.evaluation_service import EvaluationService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(checkpoint: str = "saved_models/headline_policy_tuned.pt",
         n_episodes: int = 20) -> None:
    t0 = time.time()
    lab = RoombaLab()
    env = lab.make_env()
    net = ActorCriticNet(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        actor_hidden_sizes=tuple(lab.config.get("ddpg.actor_hidden_sizes")),
        critic_hidden_sizes=tuple(lab.config.get("ddpg.critic_hidden_sizes")),
        actor_head_gain=float(lab.config.get("ddpg.actor_head_gain", 0.1)),
    )
    path = ROOT / checkpoint
    if not path.exists():
        print(f"checkpoint missing: {path}; train first")
        return
    net.load_state_dict(torch.load(path, map_location="cpu"))

    evaluator = EvaluationService(net, env)
    eps = evaluator.rollout(n_episodes=n_episodes, seed=0)
    rewards = np.array([e.reward for e in eps])
    covs = np.array([e.coverage for e in eps])
    collisions = np.array([e.collisions for e in eps])

    stats = {
        "n_episodes": n_episodes,
        "reward": {"mean": float(rewards.mean()), "median": float(np.median(rewards)),
                    "std": float(rewards.std(ddof=1)), "min": float(rewards.min()),
                    "max": float(rewards.max()),
                    "q25": float(np.quantile(rewards, 0.25)),
                    "q75": float(np.quantile(rewards, 0.75))},
        "coverage": {"mean": float(covs.mean()), "median": float(np.median(covs)),
                      "std": float(covs.std(ddof=1)), "min": float(covs.min()),
                      "max": float(covs.max()),
                      "q25": float(np.quantile(covs, 0.25)),
                      "q75": float(np.quantile(covs, 0.75))},
        "collisions_mean": float(collisions.mean()),
    }
    out_json = ROOT / "results" / "evaluation" / "per_episode.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2))

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_r.boxplot([rewards], tick_labels=["headline policy"], showmeans=True)
    ax_r.scatter([1] * len(rewards), rewards, alpha=0.5, s=30, color="#4477aa")
    ax_r.set(title=f"Per-episode reward distribution ({n_episodes} eps)",
              ylabel="episode reward")
    ax_r.grid(alpha=0.3, axis="y")

    ax_c.boxplot([covs], tick_labels=["headline policy"], showmeans=True)
    ax_c.scatter([1] * len(covs), covs, alpha=0.5, s=30, color="#cc6677")
    ax_c.set(title=f"Per-episode coverage distribution ({n_episodes} eps)",
              ylabel="coverage fraction")
    ax_c.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = ROOT / "assets" / "plots" / "per_episode_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out_json}")
    print(f"wrote {out}  ({time.time() - t0:.1f}s)")
    print(f"  mean reward = {stats['reward']['mean']:7.1f}  "
           f"median = {stats['reward']['median']:.1f}")
    print(f"  mean coverage = {stats['coverage']['mean']:.3f}  "
           f"median = {stats['coverage']['median']:.3f}")


if __name__ == "__main__":
    main()
