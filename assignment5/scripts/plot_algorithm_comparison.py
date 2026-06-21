"""Plot the DDPG vs DDPG-OU vs TD3 vs DDPG-no-replay comparison.

Bar chart with per-seed scatter points + t-distribution 95% CIs.
This is the M4 + M5 + m6 visual evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomba_lab.sdk.experiments import _t_crit_95  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "results" / "algorithms" / "comparison.json"
    if not src.exists():
        print(f"missing {src}; run scripts/run_algorithm_comparison.py first")
        sys.exit(1)
    data = json.loads(src.read_text())
    variants = dict(data["variants"])  # mutable so we can splice in true_on_policy

    # Splice in true_on_policy ablation (Layer 28) if present — closes TA NEW10
    op_path = ROOT / "results" / "algorithms" / "true_on_policy.json"
    if op_path.exists():
        op = json.loads(op_path.read_text())
        variants["ddpg_true_on_policy"] = op["rows"]

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(15, 5))
    pretty = {"ddpg_gaussian": "DDPG\n(Gaussian)",
              "ddpg_ou": "DDPG\n(OU noise)",
              "td3": "TD3\n(twin Q)",
              "ddpg_no_replay": "DDPG\n(no-update\nbuf=1, batch=128)",
              "ddpg_true_on_policy": "DDPG\n(true on-policy\nbatch=1)"}
    colours = {"ddpg_gaussian": "#4477aa", "ddpg_ou": "#117733",
                "td3": "#cc6677", "ddpg_no_replay": "#aa6699",
                "ddpg_true_on_policy": "#ddaa33"}
    labels = [pretty.get(v, v) for v in variants]
    x = np.arange(len(labels))

    for ax, key, ylabel, title in [
        (ax_r, "final_reward", "final episode reward", "Reward (mean ± 95% t-CI)"),
        (ax_c, "final_coverage", "final coverage", "Coverage (mean ± 95% t-CI)"),
    ]:
        means, cis, all_points = [], [], []
        for v_name in variants:
            vals = np.array([s[key] for s in variants[v_name]])
            n = len(vals)
            sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            means.append(vals.mean())
            cis.append(_t_crit_95(n) * sem)
            all_points.append(vals)
        bars = ax.bar(x, means, yerr=cis, color=[colours[v] for v in variants],
                       capsize=5, alpha=0.85)
        for i, pts in enumerate(all_points):
            ax.scatter([i] * len(pts), pts, color="black", s=18, zorder=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set(ylabel=ylabel, title=title)
        ax.grid(alpha=0.3, axis="y")
        _ = bars

    fig.suptitle(f"Algorithm comparison · {data['n_seeds']} seeds × "
                  f"{data['total_timesteps']} steps · primary apartment\n"
                  "DDPG (default) · OU-noise · TD3 (twin Q) · no-update "
                  "(buf=1+batch=128, tautological) · TRUE on-policy "
                  "(batch=1, Layer 28)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "assets" / "plots" / "algorithm_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
