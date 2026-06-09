"""Generate all sweep / cross-env PNG plots for the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "plots"
ASSETS.mkdir(parents=True, exist_ok=True)


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def plot_lambda_sweep_multiseed() -> None:
    d = _load("results/sweeps/lambda_multiseed.json")
    lambdas = [float(c["name"].split("=")[1]) for c in d["cells"]]
    means = [c["final_reward_mean"] for c in d["cells"]]
    cis = [c["final_reward_ci_95"] for c in d["cells"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(lambdas, means, yerr=cis, marker="o", capsize=5,
                 color="#4477aa", linewidth=2)
    ax.axvline(0.95, color="#cc6677", linestyle="--", alpha=0.6,
                label="λ = 0.95 (peak)")
    ax.set(xlabel="λ (GAE lambda)", ylabel="Final mean reward",
           title="λ-sweep — empirical bias-variance ladder (slide 16)\n3 seeds × 15k timesteps × HalfCheetah-v5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "lambda_sweep_multiseed.png", dpi=130)
    plt.close(fig)


def plot_lambda_sweep_single() -> None:
    d = _load("results/sweeps/lambda.json")
    lambdas = [float(c["name"].split("=")[1]) for c in d["cells"]]
    means = [c["final_reward_mean"] for c in d["cells"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, means, "-o", color="#4477aa", linewidth=2)
    ax.axvline(0.95, color="#cc6677", linestyle="--", alpha=0.6,
                label="λ = 0.95 (peak)")
    ax.set(xlabel="λ (GAE lambda)", ylabel="Final mean reward",
           title="λ-sweep — 1 seed × 20k timesteps (Layer 10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "lambda_sweep.png", dpi=130)
    plt.close(fig)


def plot_gamma_sweep() -> None:
    d = _load("results/sweeps/gamma.json")
    gammas = [float(c["name"].split("=")[1]) for c in d["cells"]]
    means = [c["final_reward_mean"] for c in d["cells"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gammas, means, "-o", color="#117733", linewidth=2)
    ax.set(xlabel="γ (discount)", ylabel="Final mean reward",
           title="γ-sweep — 1 seed × 20k timesteps × HalfCheetah-v5")
    fig.tight_layout()
    fig.savefig(ASSETS / "gamma_sweep.png", dpi=130)
    plt.close(fig)


def plot_clip_eps_sweep() -> None:
    d = _load("results/sweeps/clip_eps.json")
    eps = [float(c["name"].split("=")[1]) for c in d["cells"]]
    means = [c["final_reward_mean"] for c in d["cells"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps, means, "-o", color="#cc6677", linewidth=2)
    ax.set(xlabel="ε (PPO clip range)", ylabel="Final mean reward",
           title="Clip-ε sweep — 1 seed × 20k timesteps × HalfCheetah-v5")
    fig.tight_layout()
    fig.savefig(ASSETS / "clip_eps_sweep.png", dpi=130)
    plt.close(fig)


def plot_cross_env() -> None:
    d = _load("results/layer11/cross_env.json")
    fig, ax = plt.subplots(figsize=(9, 4))
    for run in d["runs"]:
        curve = run["per_iteration_reward"]
        ax.plot(range(len(curve)), curve, "-o", label=run["env_id"], linewidth=2)
    cfg = d["best_config"]
    ax.set(xlabel="Iteration", ylabel="Mean episode reward",
           title=f"Cross-env transfer — best (γ={cfg['gamma']}, λ={cfg['lambda']}, ε={cfg['clip_eps']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "cross_env_comparison.png", dpi=130)
    plt.close(fig)


def main() -> None:
    plot_lambda_sweep_multiseed()
    plot_lambda_sweep_single()
    plot_gamma_sweep()
    plot_clip_eps_sweep()
    plot_cross_env()
    print(f"wrote 5 plots to {ASSETS}")


if __name__ == "__main__":
    main()
