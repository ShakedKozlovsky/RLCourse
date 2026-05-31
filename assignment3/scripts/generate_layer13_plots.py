"""Generate plots from results/layer13/*.json for the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "layer13"
ASSETS = ROOT / "assets" / "plots"
ASSETS.mkdir(parents=True, exist_ok=True)

ACTION_LABELS = ["PUSH", "PULL", "LEGS", "CARDIO", "REST"]


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def plot_multi_seed_ci() -> None:
    d = _load("multi_seed_comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["REINFORCE", "A2C"]
    means = [d["reinforce"]["final_30pct_mean_avg"], d["a2c"]["final_30pct_mean_avg"]]
    cis = [d["reinforce"]["final_30pct_mean_ci"], d["a2c"]["final_30pct_mean_ci"]]
    ax.bar(names, means, yerr=cis, color=["#4477aa", "#cc6677"], capsize=8)
    ax.set(ylabel="Final-30 % mean reward", title="REINFORCE vs A2C — 5 seeds, 95 % CI")
    fig.tight_layout()
    fig.savefig(ASSETS / "multi_seed_ci.png", dpi=120)
    plt.close(fig)


def plot_entropy_sweep() -> None:
    d = _load("entropy_sweep")
    cells = {k: v for k, v in d.items() if not k.startswith("_")}
    bonuses = [float(k.split("=")[1]) for k in cells]
    rewards = [v["final_30pct_mean"] for v in cells.values()]
    rest_fracs = [v["action_distribution"][4] for v in cells.values()]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(bonuses, rewards, "o-", color="#4477aa", label="final-30 % reward")
    ax1.set(xlabel="Entropy bonus β", ylabel="Reward", title="A2C entropy sweep")
    ax2 = ax1.twinx()
    ax2.plot(bonuses, rest_fracs, "s--", color="#cc6677", label="REST fraction")
    ax2.set_ylabel("REST action fraction")
    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(ASSETS / "entropy_sweep.png", dpi=120)
    plt.close(fig)


def plot_reinforce_chain() -> None:
    d = _load("reinforce_chain")
    variants = ["no_baseline", "mean_baseline", "state_value_baseline_a2c"]
    labels = ["No baseline", "Mean baseline", "State-value\n(A2C)"]
    finals = [d[v]["final_30pct_mean"] for v in variants]
    stds = [d[v]["std_reward"] for v in variants]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    ax.bar(x, finals, color=["#dddddd", "#999999", "#4477aa"])
    ax.errorbar(x, finals, yerr=stds, fmt="none", color="black", capsize=6)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Final-30 % mean reward",
           title="REINFORCE → +baseline → +advantage")
    fig.tight_layout()
    fig.savefig(ASSETS / "reinforce_chain.png", dpi=120)
    plt.close(fig)


def plot_gamma_ablation() -> None:
    d = _load("gamma_ablation")
    cells = {k: v for k, v in d.items() if not k.startswith("_")}
    gammas = [float(k.split("=")[1]) for k in cells]
    rewards = [v["final_30pct_mean"] for v in cells.values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gammas, rewards, "o-", color="#4477aa")
    ax.set(xlabel="Discount factor γ", ylabel="Final-30 % mean reward",
           title="A2C — discount-factor ablation")
    fig.tight_layout()
    fig.savefig(ASSETS / "gamma_ablation.png", dpi=120)
    plt.close(fig)


def plot_masking_on_lstm() -> None:
    d = _load("masking_on_lstm_env")
    cells = {k: v for k, v in d.items() if not k.startswith("_")}
    labels = list(cells)
    rewards = [v["final_30pct_mean"] for v in cells.values()]
    stds = [v["std_reward"] for v in cells.values()]
    colors = ["#999999", "#4477aa", "#dd9988", "#cc6677"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, rewards, color=colors)
    ax.errorbar(labels, rewards, yerr=stds, fmt="none", color="black", capsize=6)
    ax.set(ylabel="Final-30 % mean reward",
           title="Masking ablation on trained LSTM dynamics")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(ASSETS / "masking_on_lstm.png", dpi=120)
    plt.close(fig)


def plot_world_model_compounding() -> None:
    """Plot world-model 1-step + multi-step rollout MSE vs baselines."""
    d = json.loads((ROOT / "results" / "layer12" / "world_model_report.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    horizons = sorted(int(k) for k, v in d["lstm_rollout_mse"].items()
                      if v is not None and not str(v).lower() == "nan")
    rollout_mses = [d["lstm_rollout_mse"][str(h)] for h in horizons]
    ax.axhline(d["persistence_one_step_mse"], color="#cc6677", linestyle="--",
                label=f'persistence (1-step) = {d["persistence_one_step_mse"]:.3f}')
    ax.axhline(d["lstm_one_step_mse"], color="#4477aa", linestyle=":",
                label=f'LSTM (1-step) = {d["lstm_one_step_mse"]:.3f}')
    ax.plot(horizons, rollout_mses, "o-", color="#117733",
             label="LSTM rollout (compounding)")
    ax.set(xlabel="Rollout horizon", ylabel="MSE",
           title="LSTM world model vs baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "world_model_compounding.png", dpi=120)
    plt.close(fig)


def plot_baselines_vs_trained() -> None:
    """Bar chart: random / round-robin / kaggle program / REINFORCE / A2C totals."""
    baselines = json.loads((ROOT / "results" / "layer12" / "baselines.json").read_text())
    multi_seed = _load("multi_seed_comparison")
    # The Layer-12 trained-policy rewards were single-seed 28-step episodes;
    # for a like-for-like comparison we use the Layer-13 multi-seed averages.
    names = [b["name"] for b in baselines] + ["REINFORCE\n(5-seed avg)",
                                                "A2C\n(5-seed avg)"]
    totals = [b["total_reward"] for b in baselines] + [
        multi_seed["reinforce"]["final_30pct_mean_avg"],
        multi_seed["a2c"]["final_30pct_mean_avg"],
    ]
    colors = ["#dddddd", "#aaaaaa", "#888888", "#4477aa", "#cc6677"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, totals, color=colors)
    ax.set(ylabel="Total / final-30 % reward",
           title="Trained agents vs baseline policies (28-day rollout)")
    fig.tight_layout()
    fig.savefig(ASSETS / "baselines_vs_trained.png", dpi=120)
    plt.close(fig)


def main() -> None:
    plot_multi_seed_ci()
    plot_entropy_sweep()
    plot_reinforce_chain()
    plot_gamma_ablation()
    plot_masking_on_lstm()
    plot_world_model_compounding()
    plot_baselines_vs_trained()
    print(f"wrote 7 Layer-13/14 figures to {ASSETS}")


if __name__ == "__main__":
    main()
