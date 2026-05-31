"""One-off: train everything end-to-end, save plots used in the README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fitness_rl.sdk.sdk import FitnessRL  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "plots"
ASSETS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "setup.json"
    cfg_data = json.loads(cfg.read_text())
    # Lower the heavy defaults for a reasonable runtime budget.
    cfg_data["world_model"]["epochs"] = 30
    cfg_data["reinforce"]["episodes"] = 60
    cfg_data["a2c"]["episodes"] = 60
    tmp = ASSETS.parent / "_runtime_config.json"
    tmp.write_text(json.dumps(cfg_data))

    sdk = FitnessRL(config_path=tmp)
    sdk.prepare_data()
    world = sdk.train_world_model()
    reinforce_hist = sdk.train_reinforce()
    a2c_hist = sdk.train_a2c()
    cmp_result = sdk.compare()

    # World-model loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(world.train_losses, label="train")
    ax.plot(world.val_losses, label="val")
    ax.set(title="LSTM world-model loss",
           xlabel="Epoch", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "world_model_loss.png", dpi=120)
    plt.close(fig)

    # REINFORCE + A2C separate reward curves
    for name, history in (("reinforce", reinforce_hist), ("a2c", a2c_hist)):
        rewards = [m.total_reward for m in history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rewards, label="total reward")
        ax.set(title=f"{name.upper()} per-episode total reward",
               xlabel="Episode", ylabel="Total reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ASSETS / f"{name}_reward.png", dpi=120)
        plt.close(fig)

    # Side-by-side comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cmp_result.reinforce_rewards, label="REINFORCE", alpha=0.85)
    ax.plot(cmp_result.a2c_rewards, label="A2C", alpha=0.85)
    ax.set(title=f"REINFORCE vs A2C (winner = {cmp_result.winner})",
           xlabel="Episode", ylabel="Total reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "comparison.png", dpi=120)
    plt.close(fig)

    # Action distribution bar chart
    labels = ["PUSH", "PULL", "LEGS", "CARDIO", "REST"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    x = np.arange(len(labels))
    ax.bar(x - width / 2, cmp_result.reinforce.action_distribution,
           width, label="REINFORCE")
    ax.bar(x + width / 2, cmp_result.a2c.action_distribution,
           width, label="A2C")
    ax.set_xticks(x, labels)
    ax.set(title="Action-distribution comparison", ylabel="Fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "action_distribution.png", dpi=120)
    plt.close(fig)

    # Dump a small summary so the README cites real numbers.
    summary = {
        "world_model_best_val_loss": world.best_val_loss,
        "world_model_best_epoch": world.best_epoch,
        "reinforce_final_reward": reinforce_hist[-1].total_reward,
        "a2c_final_reward": a2c_hist[-1].total_reward,
        "comparison": cmp_result.to_dict(),
    }
    (ASSETS / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"plots written to {ASSETS}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
