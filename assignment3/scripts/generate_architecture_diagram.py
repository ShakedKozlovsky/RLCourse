"""Audit #16: render the project architecture as a PNG for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, label, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.02",
        linewidth=1.0, edgecolor="#444444", facecolor=color,
    ))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
             ha="center", va="center", fontsize=8.5)


def _arrow(ax, xy_from, xy_to, label: str = ""):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0})
    if label:
        ax.text((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2 + 0.012,
                 label, ha="center", fontsize=7, color="#666666")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layers (top to bottom)
    # 1) Interfaces
    _box(ax, (0.04, 0.86), 0.18, 0.08, "PyQt6 GUI\n(5 tabs)", "#eef5fc")
    _box(ax, (0.30, 0.86), 0.18, 0.08, "Click CLI\n(8 commands)", "#eef5fc")
    _box(ax, (0.56, 0.86), 0.40, 0.08, "Python — sdk.sdk:FitnessRL\nsdk.evaluator:FitnessRLEvaluator",
          "#eef5fc")

    # 2) SDK facade row
    _box(ax, (0.04, 0.70), 0.45, 0.10, "sdk/sdk.py — FitnessRL\nprepare_data / train_world_model /\ntrain_reinforce / train_a2c / compare / predict",
          "#fff5e6")
    _box(ax, (0.51, 0.70), 0.45, 0.10, "sdk/evaluator.py — FitnessRLEvaluator\nevaluate_world_model / benchmark_baselines /\nqualitative_rollout",
          "#fff5e6")
    _box(ax, (0.04, 0.595), 0.45, 0.07, "sdk/env_builder.py — build_env(cfg, init, world_model)",
          "#fff5e6")

    # 3) Services layer
    services = [
        ("DataService", "#e8f5e9"),
        ("WorldModelService", "#e8f5e9"),
        ("ReinforceService", "#e8f5e9"),
        ("A2CService", "#e8f5e9"),
        ("EvaluationService", "#e8f5e9"),
        ("ComparisonService", "#e8f5e9"),
        ("WorldModelEvaluator", "#e8f5e9"),
        ("BaselinePolicies", "#e8f5e9"),
        ("ExperimentService\nExperimentStudies", "#e8f5e9"),
    ]
    cols = 5
    box_w, box_h = 0.18, 0.08
    x0, y0 = 0.025, 0.42
    gap_x, gap_y = 0.005, 0.015
    for i, (name, color) in enumerate(services):
        col, row = i % cols, i // cols
        x = x0 + col * (box_w + gap_x)
        y = y0 - row * (box_h + gap_y)
        _box(ax, (x, y), box_w, box_h, name, color)

    # 4) Models row
    _box(ax, (0.05, 0.21), 0.27, 0.08, "model/lstm_world_model.py\nLSTMWorldModel (1-layer, hidden=64)",
          "#fde7f3")
    _box(ax, (0.36, 0.21), 0.27, 0.08, "model/policy_network.py\nPolicyNet (MLP 16→128→128→5)",
          "#fde7f3")
    _box(ax, (0.67, 0.21), 0.27, 0.08, "model/actor_critic_network.py\nActorCriticNet (shared trunk + 2 heads)",
          "#fde7f3")

    # 5) Foundation row
    foundation = [
        ("environment/", "#e6e6fa", 0.05),
        ("data/", "#e6e6fa", 0.27),
        ("shared/", "#e6e6fa", 0.49),
        ("configs/setup.json", "#fff0ee", 0.71),
    ]
    for label, color, x in foundation:
        _box(ax, (x, 0.07), 0.20, 0.08, label, color)

    # Arrows
    _arrow(ax, (0.13, 0.86), (0.13, 0.80))
    _arrow(ax, (0.39, 0.86), (0.39, 0.80))
    _arrow(ax, (0.76, 0.86), (0.76, 0.80))
    _arrow(ax, (0.49, 0.70), (0.49, 0.50))  # SDK → services
    _arrow(ax, (0.49, 0.42), (0.49, 0.29))  # services → models
    _arrow(ax, (0.49, 0.21), (0.49, 0.15))  # models → foundation

    # Legend / title
    ax.text(0.5, 0.97, "fitness-rl — Architecture",
             ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.945, "Interfaces → SDK facade → Services → Models → Foundation",
             ha="center", fontsize=9, color="#666666")

    fig.savefig(OUT / "architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/architecture.png")


if __name__ == "__main__":
    main()
