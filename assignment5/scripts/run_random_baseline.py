"""Random-walk baseline — proves DDPG's training is not just luck.

For each seed, run `total_timesteps` of uniformly-random actions in [-1, 1]^2
on the same env that DDPG sees, collect episode rewards + coverage. Then we
can compare the random distribution against the trained-DDPG distribution."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from roomba_lab.sdk.sdk import RoombaLab

ROOT = Path(__file__).resolve().parents[1]


def _random_episode(env, rng) -> tuple[float, float, int]:
    obs = env.reset(seed=int(rng.integers(0, 1 << 30)))  # noqa: F841
    ep_reward = 0.0
    done = False
    info = {"coverage": 0.0, "step": 0}
    while not done:
        a = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
        _, r, done, info = env.step(a)
        ep_reward += r
    return ep_reward, info["coverage"], info["step"]


def main(n_seeds: int = 5, n_episodes_per_seed: int = 10) -> None:
    started = time.time()
    lab = RoombaLab()
    results = []
    for seed in range(n_seeds):
        env = lab.make_env()
        rng = np.random.default_rng(seed)
        ep_rewards, ep_covs, ep_lens = [], [], []
        for _ in range(n_episodes_per_seed):
            r, c, length = _random_episode(env, rng)
            ep_rewards.append(r)
            ep_covs.append(c)
            ep_lens.append(length)
        results.append({"seed": seed,
                        "mean_reward": float(np.mean(ep_rewards)),
                        "mean_coverage": float(np.mean(ep_covs)),
                        "mean_length": float(np.mean(ep_lens))})
    out = ROOT / "results" / "baselines" / "random.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rewards = np.array([r["mean_reward"] for r in results])
    covs = np.array([r["mean_coverage"] for r in results])
    payload = {
        "kind": "random",
        "n_seeds": n_seeds,
        "n_episodes_per_seed": n_episodes_per_seed,
        "per_seed": results,
        "agg": {
            "mean_reward": float(rewards.mean()),
            "ci95_reward": float(1.96 * rewards.std(ddof=1) / np.sqrt(n_seeds)),
            "mean_coverage": float(covs.mean()),
            "ci95_coverage": float(1.96 * covs.std(ddof=1) / np.sqrt(n_seeds)),
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  ({time.time() - started:.1f}s)")
    print(f"  random baseline:  reward={payload['agg']['mean_reward']:7.2f} "
           f"± {payload['agg']['ci95_reward']:.2f}  "
           f"coverage={payload['agg']['mean_coverage']:.3f} "
           f"± {payload['agg']['ci95_coverage']:.3f}")


if __name__ == "__main__":
    main()
