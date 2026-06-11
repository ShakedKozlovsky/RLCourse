"""Run all three sweeps (λ, γ, clip-ε) and save JSON to results/sweeps/.

Tuned for a ~5-10 min CPU budget. Bigger numbers in configs/setup.json:experiments
let you re-run at the full PRD-stated budget.
"""

from __future__ import annotations

import time
from pathlib import Path

from proximal_lab.sdk.sdk import ProximalLab
from proximal_lab.sdk.experiments import ExperimentService

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg_path = ROOT / "configs" / "setup.json"
    sdk = ProximalLab(config_path=cfg_path)
    # 20k timesteps × 1 seed per cell — enough to differentiate cells
    # without dominating wall-clock; bump n_seeds=3 for the final run.
    svc = ExperimentService(sdk, timesteps_per_cell=20000, n_seeds=1,
                              steps_per_rollout=1024)
    out_dir = ROOT / "results" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, runner in (
        ("lambda", svc.run_lambda_sweep),
        ("gamma", svc.run_gamma_sweep),
        ("clip_eps", svc.run_clip_eps_sweep),
    ):
        t = time.time()
        print(f"[{kind}] starting …")
        report = runner()
        ExperimentService.save(report, out_dir / f"{kind}.json")
        print(f"  wrote {out_dir / (kind + '.json')}  ({time.time() - t:.1f}s)")
        for cell in report.cells:
            print(f"    {cell.name}: final={cell.final_reward_mean:.2f} "
                   f"± {cell.final_reward_ci_95:.2f}")


if __name__ == "__main__":
    main()
