"""Layer-15 plots: 3-algo learning curves, baselines bar chart, REINFORCE→A2C→PPO chain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "plots"
ASSETS.mkdir(parents=True, exist_ok=True)


def _load() -> dict:
    return json.loads((ROOT / "results" / "layer15" / "full_budget_multiseed.json").read_text())


def plot_three_algo_curves() -> None:
    d = _load()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"reinforce": "#4477aa", "a2c": "#cc6677", "ppo": "#117733"}
    for algo, r in d["results"].items():
        arr = np.array(r["per_seed_rewards"])  # shape (seeds, episodes)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(arr.shape[1])
        ax.plot(x, mean, color=colors[algo], label=f"{algo.upper()}")
        ax.fill_between(x, mean - std, mean + std, color=colors[algo], alpha=0.15)
    ax.set(xlabel="Episode", ylabel="Total reward",
           title="REINFORCE vs A2C vs PPO — 3 seeds × 300 episodes (mean ± 1 σ)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "three_algo_curves.png", dpi=120)
    plt.close(fig)


def plot_three_algo_final_ci() -> None:
    d = _load()
    fig, ax = plt.subplots(figsize=(7, 4))
    algos = ["REINFORCE", "A2C", "PPO"]
    keys = ["reinforce", "a2c", "ppo"]
    means = [d["results"][k]["final_30pct_mean_avg"] for k in keys]
    cis = [d["results"][k]["final_30pct_mean_ci_95"] for k in keys]
    colors = ["#4477aa", "#cc6677", "#117733"]
    ax.bar(algos, means, yerr=cis, color=colors, capsize=8)
    ax.set(ylabel="Final-30 % mean reward",
           title="3-algo chain — 300 episodes × 3 seeds, 95 % CI")
    fig.tight_layout()
    fig.savefig(ASSETS / "three_algo_final_ci.png", dpi=120)
    plt.close(fig)


def plot_baselines_vs_trained_post_fix() -> None:
    d = _load()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bnames = list(d["baselines_first_seed"])
    btotals = [d["baselines_first_seed"][n]["total"] for n in bnames]
    keys = ["reinforce", "a2c", "ppo"]
    tnames = [k.upper() + "\n(3-seed avg)" for k in keys]
    ttotals = [d["results"][k]["final_30pct_mean_avg"] for k in keys]
    names = bnames + tnames
    totals = btotals + ttotals
    colors = (["#dddddd", "#aaaaaa", "#888888"]
               + ["#4477aa", "#cc6677", "#117733"])
    ax.bar(names, totals, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(ylabel="Reward over 28-day rollout",
           title="Baselines vs trained agents — post-reward-fix")
    fig.tight_layout()
    fig.savefig(ASSETS / "baselines_vs_trained_post_fix.png", dpi=120)
    plt.close(fig)


def main() -> None:
    plot_three_algo_curves()
    plot_three_algo_final_ci()
    plot_baselines_vs_trained_post_fix()
    print(f"wrote 3 Layer-15 figures to {ASSETS}")


if __name__ == "__main__":
    main()
