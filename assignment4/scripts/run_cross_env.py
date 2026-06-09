"""Layer 11: take the best (γ, λ, clip-ε) trio from Layer 10 and re-run on Walker2d-v5.

Best from Layer 10's HalfCheetah-v5 sweeps:
    γ = 0.999, λ = 0.95, clip_eps = 0.3
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.services.experiment_service import _write_temp_cfg

ROOT = Path(__file__).resolve().parents[1]


def _train_one(base_cfg: dict, env_id: str, total_ts: int, seed: int) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["env"]["gamma"] = 0.999
    cfg["gae"]["lambda"] = 0.95
    cfg["ppo"]["clip_eps"] = 0.3
    sdk = ProximalLab(config_path=_write_temp_cfg(cfg))
    t = time.time()
    result = sdk.train_ppo(env_id=env_id, total_timesteps=total_ts,
                            steps_per_rollout=1024, seed=seed)
    return {
        "env_id": env_id,
        "seed": seed,
        "final_mean_reward": float(result.final_mean_reward),
        "total_timesteps": result.total_timesteps,
        "n_iterations": len(result.diagnostics),
        "wall_clock_s": round(time.time() - t, 1),
        "per_iteration_reward": [d.mean_episode_reward for d in result.diagnostics],
    }


def main() -> None:
    cfg_path = ROOT / "configs" / "setup.json"
    base_cfg = json.loads(cfg_path.read_text())
    total_ts = 30000
    out = {"best_config": {"gamma": 0.999, "lambda": 0.95, "clip_eps": 0.3},
            "runs": []}
    for env_id in ("HalfCheetah-v5", "Walker2d-v5"):
        print(f"[{env_id}] training with best config …")
        record = _train_one(base_cfg, env_id, total_ts=total_ts, seed=0)
        out["runs"].append(record)
        print(f"  final={record['final_mean_reward']:.2f} "
               f"({record['wall_clock_s']}s)")
    out_dir = ROOT / "results" / "layer11"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_env.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir}/cross_env.json")


if __name__ == "__main__":
    main()
