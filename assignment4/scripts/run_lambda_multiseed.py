"""Layer-13 audit fix #1: re-run the headline λ-sweep with **3 seeds** for real CIs.

Layer 10 ran 1 seed per cell, producing CI=0. To claim the inverted-U around
λ=0.95 is statistically real, we need ≥ 3 seeds.
"""

from __future__ import annotations

import time
from pathlib import Path

from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.services.experiment_service import ExperimentService

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sdk = ProximalLab(config_path=ROOT / "configs" / "setup.json")
    svc = ExperimentService(sdk, timesteps_per_cell=15000, n_seeds=3,
                              steps_per_rollout=1024)
    print("[λ-sweep × 3 seeds × 6 cells × 15k timesteps] starting …")
    t = time.time()
    report = svc.run_lambda_sweep()
    out = ROOT / "results" / "sweeps" / "lambda_multiseed.json"
    ExperimentService.save(report, out)
    print(f"wrote {out}  ({time.time() - t:.1f}s)")
    for cell in report.cells:
        print(f"  {cell.name}: {cell.final_reward_mean:+.2f} ± {cell.final_reward_ci_95:.2f}")


if __name__ == "__main__":
    main()
