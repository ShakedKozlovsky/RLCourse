"""Run Layer-12 diagnostics on the real 84-day Kaggle trajectory.

* WorldModelEvaluator: persistence + linear vs LSTM, horizons {1, 7, 28}
* Baseline policies: random / round-robin / Kaggle-program
* Qualitative rollout: 28-day greedy schedule with per-step reward decomposition
"""

from __future__ import annotations

import json
from pathlib import Path

from fitness_rl.sdk.evaluator import FitnessRLEvaluator
from fitness_rl.sdk.sdk import FitnessRL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "layer12"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = ROOT / "configs" / "setup.json"
    cfg_data = json.loads(cfg.read_text())
    cfg_data["world_model"]["epochs"] = 30
    cfg_data["reinforce"]["episodes"] = 60
    cfg_data["a2c"]["episodes"] = 60
    tmp = OUT / "_runtime_config.json"
    tmp.write_text(json.dumps(cfg_data))

    sdk = FitnessRL(config_path=tmp)
    sdk.prepare_data()
    sdk.train_world_model()
    sdk.train_reinforce()
    sdk.train_a2c()
    ev = FitnessRLEvaluator(sdk)

    # 1) World-model evaluation
    wm = ev.evaluate_world_model(horizons=(1, 7, 28))
    (OUT / "world_model_report.json").write_text(json.dumps(wm.to_dict(), indent=2))

    # 2) Baselines
    baselines = ev.benchmark_baselines()
    (OUT / "baselines.json").write_text(json.dumps(
        [{"name": b.name, "total_reward": b.total_reward,
          "action_distribution": b.action_distribution} for b in baselines],
        indent=2,
    ))

    # 3) Qualitative greedy rollouts (28-day) for both algos
    for algo in ("reinforce", "a2c"):
        traj = ev.qualitative_rollout(algo=algo)
        (OUT / f"qualitative_{algo}.txt").write_text(traj.as_table())
        (OUT / f"qualitative_{algo}.json").write_text(json.dumps(
            {"steps": [{"step": s.step, "action": s.action,
                        "action_name": s.action_name,
                        "gain": s.gain, "overload": s.overload,
                        "imbalance": s.imbalance,
                        "total_reward": s.total_reward}
                       for s in traj.steps],
             "total_reward": traj.total_reward}, indent=2,
        ))

    tmp.unlink(missing_ok=True)
    print(f"diagnostics written to {OUT}")


if __name__ == "__main__":
    main()
