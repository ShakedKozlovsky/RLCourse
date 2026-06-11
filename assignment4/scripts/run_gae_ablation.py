"""GAE-as-advantage-quality ablation — direct test of slide-14's claim.

Slide 14: *"Advantage quality determines PPO stability"*. The headline λ-sweep
showed final-reward differences across λ; this ablation goes further and tests
the **within-run stability** claim by measuring:

  - per-iteration reward variance (the rollout-to-rollout swing)
  - explained-variance trajectory (critic fit quality)
  - mean KL per iteration (proxy for "is the policy moving sensibly")

across three advantage estimators (λ = 0, 0.95, 1.0) on the *same* env, with
3 seeds. The slide-14 prediction:

  λ=0  (TD-only):       low variance per step, high bias → choppy critic fit
  λ=0.95 (GAE):         balanced bias/variance → stable + good policy
  λ=1  (MC-only):       no bias but high variance → choppy reward curves
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from proximal_lab.sdk.sdk import ProximalLab  # noqa: E402
from proximal_lab.sdk.experiments import _write_temp_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "layer16_gae_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
TOTAL_TS = 15000
SPR = 1024
LAMBDAS = (0.0, 0.95, 1.0)


def _run_one(base_cfg: dict, lam: float, seed: int) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["gae"]["lambda"] = float(lam)
    cfg_path = _write_temp_cfg(cfg)
    sdk = ProximalLab(config_path=cfg_path)
    result = sdk.train_ppo(env_id="HalfCheetah-v5", total_timesteps=TOTAL_TS,
                            steps_per_rollout=SPR, seed=seed)
    return {
        "rewards": [d.mean_episode_reward for d in result.diagnostics],
        "explained_variance": [d.explained_variance for d in result.diagnostics],
        "mean_kl": [d.mean_kl for d in result.diagnostics],
        "clip_fraction": [d.clip_fraction for d in result.diagnostics],
    }


def _aggregate(per_seed_traj: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std across seeds, per iteration. Variable lengths → truncate to min."""
    min_len = min(len(t) for t in per_seed_traj)
    arr = np.array([t[:min_len] for t in per_seed_traj], dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "setup.json").read_text())
    started = time.time()
    raw: dict[float, list[dict]] = {lam: [] for lam in LAMBDAS}
    for lam in LAMBDAS:
        for seed in SEEDS:
            print(f"[λ={lam}, seed={seed}] training …")
            t = time.time()
            raw[lam].append(_run_one(base_cfg, lam, seed))
            print(f"  ({time.time() - t:.1f}s)")
    # Save raw + plot
    payload = {str(lam): raw[lam] for lam in LAMBDAS}
    (OUT_DIR / "gae_ablation_raw.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_DIR / 'gae_ablation_raw.json'}  "
           f"(total runtime {time.time() - started:.1f}s)")

    # Three-panel figure: reward, explained variance, KL
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colours = {0.0: "#cc6677", 0.95: "#4477aa", 1.0: "#117733"}
    labels = {0.0: "λ=0.0 (TD-only)", 0.95: "λ=0.95 (GAE)",
               1.0: "λ=1.0 (MC-only)"}
    for ax, (key, title, ylabel) in zip(
        axes,
        [("rewards", "Reward — Loop pillar", "mean episode reward"),
         ("explained_variance", "Critic fit — Signal pillar", "explained variance"),
         ("mean_kl", "Policy step — Policy pillar", "mean KL per update")],
        strict=True,
    ):
        for lam in LAMBDAS:
            trajs = [r[key] for r in raw[lam]]
            mean, std = _aggregate(trajs)
            x = np.arange(len(mean))
            ax.plot(x, mean, "-", color=colours[lam], label=labels[lam], linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=colours[lam], alpha=0.18)
        ax.set(title=title, xlabel="iteration", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("GAE advantage-quality ablation — slide-14 stability pillars across λ\n"
                  "3 seeds × HalfCheetah-v5 × 15k timesteps; bands = ±1σ across seeds",
                  fontsize=11)
    fig.tight_layout()
    out = ROOT / "assets" / "plots" / "gae_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
