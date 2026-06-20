"""Run the noise-σ sweep — reflection-Q2 evidence script.

Cells = {0.0, 0.1, 0.2, 0.4} × 3 seeds. The σ=0.0 cell is the "no exploration"
baseline that the reflection-Q2 answer leans on."""

from __future__ import annotations

import time
from pathlib import Path

from roomba_lab.sdk.experiments import ExperimentService
from roomba_lab.sdk.sdk import RoombaLab

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    started = time.time()
    lab = RoombaLab(config_path=ROOT / "configs" / "setup.json")
    svc = ExperimentService(lab, n_seeds=3, total_timesteps=4000)
    out = svc.run("noise_sigma")
    agg = ExperimentService.aggregate(out)
    print(f"wrote {out}  ({time.time() - started:.1f}s)")
    for cell, m in agg.items():
        print(f"  σ={cell}  reward={m['mean_reward']:7.2f} ± {m['ci95_reward']:.2f}"
               f"  coverage={m['mean_coverage']:.3f}")


if __name__ == "__main__":
    main()
