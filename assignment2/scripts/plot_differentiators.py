"""Plotting helpers for the three differentiator analyses.

Separated from ``run_differentiators.py`` to keep both files under 150 LOC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dqn_trader.shared.types import Action  # noqa: E402

OUT = Path("assets/plots")


def plot_window_sweep(result) -> None:
    """Bar chart of test return + Sharpe across window sizes."""
    names = [c.name for c in result.conditions]
    returns = [c.metrics.total_return for c in result.conditions]
    sharpes = [c.metrics.sharpe for c in result.conditions]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    ax1.bar(names, returns, color="#4C72B0")
    ax1.set_title("Total return by window size")
    ax1.set_ylabel("total return")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(returns):
        ax1.text(i, v, f"{v:.2%}", ha="center", va="bottom", fontsize=8)
    ax2.bar(names, sharpes, color="#DD8452")
    ax2.set_title("Sharpe ratio by window size")
    ax2.set_ylabel("annualised Sharpe")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(sharpes):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / "window_sensitivity.png", dpi=120)
    plt.close(fig)


def plot_action_distribution(train_d, test_d) -> None:
    """Side-by-side bar chart of train vs test action fractions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    labels = ["Sell", "Hold", "Buy"]
    colors = ["#C44E52", "#8C8C8C", "#4C72B0"]
    train_fracs = [train_d.sell_frac, train_d.hold_frac, train_d.buy_frac]
    test_fracs = [test_d.sell_frac, test_d.hold_frac, test_d.buy_frac]
    ax1.bar(labels, train_fracs, color=colors)
    ax1.set_title(f"Train action dist. (n={train_d.total_steps})")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(train_fracs):
        ax1.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax2.bar(labels, test_fracs, color=colors)
    ax2.set_title(f"Test action dist. (n={test_d.total_steps})")
    ax2.set_ylabel("fraction")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(test_fracs):
        ax2.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    fig.savefig(OUT / "action_distribution.png", dpi=120)
    plt.close(fig)


def plot_qvalue_heatmap(snap) -> None:
    """Q-value curves + buy/sell markers + portfolio value aligned below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), tight_layout=True, sharex=True)
    steps = np.arange(len(snap.q_sell))
    ax1.plot(steps, snap.q_sell, label="Q(Sell)", alpha=0.8)
    ax1.plot(steps, snap.q_hold, label="Q(Hold)", alpha=0.8)
    ax1.plot(steps, snap.q_buy, label="Q(Buy)", alpha=0.8)
    buy_mask = snap.actions_taken == Action.BUY
    sell_mask = snap.actions_taken == Action.SELL
    ax1.scatter(
        steps[buy_mask],
        snap.q_buy[buy_mask],
        marker="^",
        c="green",
        s=40,
        zorder=5,
        label="Buy taken",
    )
    ax1.scatter(
        steps[sell_mask],
        snap.q_sell[sell_mask],
        marker="v",
        c="red",
        s=40,
        zorder=5,
        label="Sell taken",
    )
    ax1.set_ylabel("Q-value")
    ax1.set_title("Q-values over test slice (greedy policy)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(steps, snap.portfolio_values[: len(steps)], color="#4C72B0", linewidth=2)
    ax2.set_xlabel("test step (day)")
    ax2.set_ylabel("portfolio value ($)")
    ax2.set_title("Portfolio value during Q-value rollout")
    ax2.grid(alpha=0.3)
    fig.savefig(OUT / "qvalue_heatmap.png", dpi=120)
    plt.close(fig)
