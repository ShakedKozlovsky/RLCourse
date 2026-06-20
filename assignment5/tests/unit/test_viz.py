"""Layer 12 — visualisation plot generators (PNG files exist + non-empty)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roomba_lab.shared.types import StepDiagnostic, TrainResult
from roomba_lab.tools.viz.plots import (
    plot_coverage_heatmap,
    plot_critic_loss,
    plot_learning_curve,
    plot_trajectory_overlay,
)


def _fake_result(n: int = 10) -> TrainResult:
    r = TrainResult()
    r.diagnostics = [
        StepDiagnostic(step=i * 10, actor_loss=0.1 - 0.01 * i,
                       critic_loss=1.0 - 0.05 * i,
                       mean_q=0.0, sigma=0.2, episode_reward=float(i * 2),
                       coverage=float(i) / n)
        for i in range(n)
    ]
    return r


def test_learning_curve_png(tmp_path: Path) -> None:
    p = tmp_path / "lc.png"
    plot_learning_curve(_fake_result(), p)
    assert p.exists() and p.stat().st_size > 1000


def test_critic_loss_png(tmp_path: Path) -> None:
    p = tmp_path / "cl.png"
    plot_critic_loss(_fake_result(), p)
    assert p.exists() and p.stat().st_size > 1000


def test_trajectory_overlay_png(tmp_path: Path) -> None:
    p = tmp_path / "traj.png"
    plot_trajectory_overlay([(0, 0), (1, 0), (1, 1), (0.5, 0.7)],
                              [(0, 0), (2, 0), (2, 2), (0, 2)], p)
    assert p.exists() and p.stat().st_size > 1000


def test_coverage_heatmap_png(tmp_path: Path) -> None:
    p = tmp_path / "h.png"
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[5:15, 5:15] = 1
    grid[0, :] = 255
    plot_coverage_heatmap(grid, p)
    assert p.exists() and p.stat().st_size > 1000
