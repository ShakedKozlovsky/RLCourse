"""Layer 15: full-budget multi-seed run including PPO + 3-way comparison.

Trains REINFORCE, A2C, and PPO at the PRD-stated **300-episode budget** across
3 seeds (5 would 5× the runtime; 3 is a sensible budget for the extended chain)
and dumps per-algorithm reward arrays + summary statistics for the README.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from fitness_rl.sdk.evaluator import FitnessRLEvaluator
from fitness_rl.sdk.sdk import FitnessRL
from fitness_rl.services.experiment_base import write_temp_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "layer15"
OUT.mkdir(parents=True, exist_ok=True)


def _aggregate(reward_arrays: list[list[float]]) -> dict:
    arr = np.array(reward_arrays)
    finals = arr[:, -max(1, int(0.3 * arr.shape[1])):].mean(axis=1)
    means = arr.mean(axis=1)
    return {
        "n_seeds": int(arr.shape[0]),
        "n_episodes": int(arr.shape[1]),
        "per_seed_rewards": [list(r) for r in arr],
        "final_30pct_mean_avg": float(finals.mean()),
        "final_30pct_mean_ci_95": float(1.96 * finals.std(ddof=1)
                                          / np.sqrt(finals.size)) if finals.size > 1 else 0.0,
        "overall_mean_avg": float(means.mean()),
    }


def main() -> None:
    cfg_base = json.loads((ROOT / "configs" / "setup.json").read_text())
    seeds = (0, 1, 2)
    episodes = 300
    started = time.time()
    runs: dict[str, list[list[float]]] = {
        "reinforce": [], "a2c": [], "ppo": [],
    }
    baselines_seed = {}
    qualitative = {}
    for seed in seeds:
        cfg = deepcopy(cfg_base)
        cfg["seed"] = int(seed)
        tmp = write_temp_cfg(cfg)
        sdk = FitnessRL(config_path=tmp)
        sdk.prepare_data()
        sdk.train_world_model()
        for algo, train in (
            ("reinforce", sdk.train_reinforce),
            ("a2c",       sdk.train_a2c),
            ("ppo",       sdk.train_ppo),
        ):
            t = time.time()
            history = train(episodes=episodes)
            rewards = [m.total_reward for m in history]
            runs[algo].append(rewards)
            print(f"  seed {seed} {algo}: {rewards[-1]:.3f} "
                  f"final-30%={np.mean(rewards[-90:]):.3f} ({time.time()-t:.1f}s)")
        if seed == seeds[0]:
            ev = FitnessRLEvaluator(sdk)
            for b in ev.benchmark_baselines():
                baselines_seed[b.name] = {"total": b.total_reward,
                                            "action_distribution": b.action_distribution}
            for algo in ("reinforce", "a2c", "ppo"):
                traj = ev.qualitative_rollout(algo=algo)
                qualitative[algo] = {
                    "total_reward": traj.total_reward,
                    "actions": [s.action_name for s in traj.steps],
                }
    payload = {
        "runtime_seconds": round(time.time() - started, 1),
        "episodes_per_algo_per_seed": episodes,
        "results": {k: _aggregate(v) for k, v in runs.items()},
        "baselines_first_seed": baselines_seed,
        "qualitative_first_seed": qualitative,
    }
    (OUT / "full_budget_multiseed.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}/full_budget_multiseed.json")


if __name__ == "__main__":
    main()
