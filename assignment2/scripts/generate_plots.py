"""Render the comparison plots from experiment outputs into assets/plots/.

Consumes:
    results/run_<ts>/metrics.csv     — per-episode training metrics
    results/run_<ts>/checkpoints/*   — best/last checkpoints
    results/<experiment_name>.json   — ExperimentService output payload
    results/backtest/*.npz / *.json  — per-backtest curves and metrics

Usage:
    uv run python scripts/generate_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def latest_metrics_csv() -> Path | None:
    runs = sorted(RESULTS.glob("run_*/metrics.csv"))
    return runs[-1] if runs else None


def plot_training_curves(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), tight_layout=True)
    df.plot(x="episode", y="reward", ax=axes[0, 0], legend=False, title="Episode reward")
    df.plot(x="episode", y="loss", ax=axes[0, 1], legend=False, title="Mean optimization loss")
    df.plot(x="episode", y="epsilon", ax=axes[1, 0], legend=False, title="ε-greedy schedule")
    df.plot(x="episode", y="val_return", ax=axes[1, 1], legend=False, title="Validation return")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.savefig(OUT / "training_curves.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT/'training_curves.png'}")


def plot_experiment(name: str) -> None:
    payload_path = RESULTS / f"{name}.json"
    if not payload_path.exists():
        print(f"skip {name} — no results yet")
        return
    payload = json.loads(payload_path.read_text())
    conditions = payload["conditions"]
    metrics = ["total_return", "sharpe", "max_drawdown", "win_rate", "n_trades"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.4), tight_layout=True)
    names = [c["name"] for c in conditions]
    for ax, m in zip(axes, metrics, strict=True):
        vals = [c["metrics"][m] for c in conditions]
        ax.bar(names, vals, color=("#4C72B0", "#DD8452"))
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom")
    fig.suptitle(f"experiment: {name}")
    fig.savefig(OUT / f"experiment_{name}.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT/f'experiment_{name}.png'}")


def plot_equity_overlay(experiment: str, conditions: list[str]) -> None:
    """Equity vs Buy-and-Hold overlay for the two conditions of one experiment."""
    bt_dir = RESULTS / "backtest"
    fig, ax = plt.subplots(figsize=(9, 4), tight_layout=True)
    for cond in conditions:
        npz_path = bt_dir / f"{experiment}__{cond}.npz"
        if not npz_path.exists():
            print(f"skip {npz_path.name} — not found")
            continue
        d = np.load(npz_path)
        ax.plot(d["equity"], label=f"DQN — {cond}", linewidth=2)
    # Use the first available benchmark (Buy-and-Hold same for both AAPL conditions).
    for cond in conditions:
        npz_path = bt_dir / f"{experiment}__{cond}.npz"
        if npz_path.exists():
            d = np.load(npz_path)
            ax.plot(d["benchmark"], label=f"Buy & Hold — {cond}", linestyle="--", alpha=0.6)
            break
    ax.set_xlabel("test-slice step (day)")
    ax.set_ylabel("portfolio value ($)")
    ax.set_title(f"Backtest equity — {experiment}")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = OUT / f"equity_{experiment}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    csv = latest_metrics_csv()
    if csv:
        plot_training_curves(csv)
    experiments = {
        "dqn_vs_dueling": ["vanilla_dqn", "dueling_dqn"],
        "uniform_vs_per": ["uniform_replay", "prioritized_replay"],
        "reward_variants": ["baseline", "risk_adjusted"],
        "cross_ticker": ["AAPL", "SPY"],
    }
    for name, conds in experiments.items():
        plot_experiment(name)
        plot_equity_overlay(name, conds)


if __name__ == "__main__":
    main()
