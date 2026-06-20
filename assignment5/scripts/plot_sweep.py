"""Plot a sweep JSON as a bar chart of mean ± 95% CI per cell."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomba_lab.sdk.experiments import ExperimentService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(kind: str = "noise_sigma") -> None:
    src = ROOT / "results" / "sweeps" / f"{kind}.json"
    if not src.exists():
        print(f"missing {src}; run the matching scripts/run_*_sweep.py first")
        return
    agg = ExperimentService.aggregate(src)
    labels = sorted(agg.keys(), key=lambda s: float(s))
    means = [agg[c]["mean_reward"] for c in labels]
    cis = [agg[c]["ci95_reward"] for c in labels]
    covs = [agg[c]["mean_coverage"] for c in labels]
    payload = json.loads(src.read_text())

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    ax_r.bar(x, means, yerr=cis, color="#4477aa", capsize=4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels)
    ax_r.set(xlabel=kind, ylabel="final episode reward (mean ± 95% CI)",
              title=f"{kind} sweep — reward")
    ax_r.grid(alpha=0.3, axis="y")

    ax_c.bar(x, covs, color="#cc6677")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set(xlabel=kind, ylabel="mean final coverage",
              title=f"{kind} sweep — coverage")
    ax_c.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{payload['kind']} sweep · {payload['n_seeds']} seeds × "
                  f"{payload['total_timesteps']} steps", fontsize=11)
    fig.tight_layout()
    out = ROOT / "assets" / "plots" / f"{kind}_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "noise_sigma")
