"""Headless GUI smoke tests under ``QT_QPA_PLATFORM=offscreen``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from proximal_lab.interface.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def sdk_config(tmp_path: Path) -> Path:
    cfg = {
        "version": "1.00",
        "seed": 0,
        "env": {"id": "HalfCheetah-v5", "gamma": 0.99, "n_parallel_envs": 2,
                 "secondary_id": "Walker2d-v5", "max_episode_steps": 1000},
        "gae": {"lambda": 0.95},
        "actor_critic": {"hidden_sizes": [32, 32], "activation": "tanh",
                          "shared_trunk": False,
                          "log_std_init": -0.5, "log_std_min": -5.0, "log_std_max": 2.0},
        "ppo": {"total_timesteps": 256, "steps_per_rollout": 128,
                 "minibatch_size": 16, "n_epochs_per_update": 1,
                 "clip_eps": 0.2, "lr": 3e-4, "value_coef": 0.5,
                 "entropy_coef": 0.0, "max_grad_norm": 0.5, "target_kl_stop": None},
        "paths": {"results_dir": str(tmp_path / "results"),
                   "assets_dir": str(tmp_path / "assets"),
                   "checkpoints_dir": str(tmp_path / "saved_models"),
                   "wiki_dir": str(tmp_path / "wiki")},
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(cfg))
    return path


def test_main_window_constructs(qapp, sdk_config: Path) -> None:  # noqa: ARG001
    window = MainWindow(config_path=sdk_config)
    assert "proximal-lab" in window.windowTitle()
    assert window.tabs.count() == 3


def test_tab_labels(qapp, sdk_config: Path) -> None:  # noqa: ARG001
    window = MainWindow(config_path=sdk_config)
    labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert labels == ["Train", "Sweep", "Compare"]


def test_plot_widget_draws(qapp) -> None:  # noqa: ARG001
    from proximal_lab.interface.gui.plot_widget import PlotWidget

    widget = PlotWidget()
    widget.draw(lambda ax: ax.plot([1, 2, 3]))  # type: ignore[attr-defined]


def test_sweep_tab_missing_file_status(qapp, sdk_config: Path) -> None:  # noqa: ARG001
    window = MainWindow(config_path=sdk_config)
    sweep_tab = window.tabs.widget(1)
    # The tab has a Plot button; clicking it without sweep files should show
    # "missing" status without crashing
    sweep_tab._on_plot()  # noqa: SLF001 — direct test of GUI handler
