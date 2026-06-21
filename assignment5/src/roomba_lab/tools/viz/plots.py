"""Static-PNG plot generators — the two mandatory spec graphs plus extras."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roomba_lab.shared.types import TrainResult  # noqa: E402


def plot_learning_curve(result: TrainResult, out: Path,
                         title: str = "DDPG Learning Curve") -> None:
    """Plot learning curve."""
    steps = [d.step for d in result.diagnostics]
    rewards = [d.episode_reward for d in result.diagnostics]
    coverage = [d.coverage for d in result.diagnostics]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, rewards, color="#4477aa", label="episode reward", linewidth=2)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("episode reward", color="#4477aa")
    ax1.tick_params(axis="y", labelcolor="#4477aa")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps, coverage, color="#cc6677", label="coverage", linewidth=2,
              alpha=0.7)
    ax2.set_ylabel("coverage fraction", color="#cc6677")
    ax2.tick_params(axis="y", labelcolor="#cc6677")
    plt.title(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_critic_loss(result: TrainResult, out: Path,
                      title: str = "Critic Loss vs Training Step") -> None:
    """Plot critic loss."""
    steps = [d.step for d in result.diagnostics]
    losses = [d.critic_loss for d in result.diagnostics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses, color="#cc6677", linewidth=2)
    ax.set(xlabel="training step", ylabel="critic loss (MSE)", title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_trajectory_overlay(robot_trajectory: list[tuple[float, float]],
                             polygon_vertices: list[tuple[float, float]],
                             out: Path, title: str = "Robot trajectory overlay") -> None:
    """Plot trajectory overlay."""
    fig, ax = plt.subplots(figsize=(7, 7))
    poly = np.array(polygon_vertices)
    ax.fill(poly[:, 0], poly[:, 1], color="#eaeaea", alpha=0.6,
             edgecolor="black", linewidth=1.5)
    if robot_trajectory:
        xs = [p[0] for p in robot_trajectory]
        ys = [p[1] for p in robot_trajectory]
        ax.plot(xs, ys, color="#4477aa", linewidth=1.5, alpha=0.9, label="trajectory")
        ax.scatter([xs[0]], [ys[0]], color="green", s=80, label="start", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], color="red", s=80, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_coverage_heatmap(grid: np.ndarray, out: Path,
                           title: str = "Coverage heatmap") -> None:
    """grid: 0=unvisited, 1=visited, 255=obstacle."""
    display = grid.astype(np.float32)
    display[grid == 255] = np.nan
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(display, origin="lower", cmap="viridis",
                    interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="visited (1) / unvisited (0)")
    ax.set(title=title, xlabel="grid col", ylabel="grid row")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
