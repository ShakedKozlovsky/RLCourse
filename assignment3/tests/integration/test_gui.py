"""Headless GUI smoke tests — runs under ``QT_QPA_PLATFORM=offscreen``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force the offscreen Qt platform *before* importing PyQt6 below.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from fitness_rl.interface.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Module-scoped QApplication — QApplication must be a singleton."""
    app = QApplication.instance() or QApplication([])
    yield app  # type: ignore[misc]


def test_main_window_constructs(qapp: QApplication, sdk_config: Path) -> None:
    window = MainWindow(config_path=sdk_config)
    assert window.windowTitle() == "fitness-rl — REINFORCE + A2C"
    assert window.tabs.count() == 5


def test_main_window_tab_labels(qapp: QApplication, sdk_config: Path) -> None:
    window = MainWindow(config_path=sdk_config)
    labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert labels == ["Data", "World model", "REINFORCE", "A2C", "Compare"]


def test_main_window_shares_sdk_instance(qapp: QApplication, sdk_config: Path) -> None:
    window = MainWindow(config_path=sdk_config)
    assert window.sdk is not None
    # Trigger prepare_data via the SDK — every tab references the same SDK.
    out = window.sdk.prepare_data()
    assert out.states.shape[1] == 16


def test_plot_widget_draws_without_error(qapp: QApplication, sdk_config: Path) -> None:
    from fitness_rl.interface.gui.plot_widget import PlotWidget

    widget = PlotWidget()
    widget.draw(lambda ax: ax.plot([1, 2, 3]))  # type: ignore[attr-defined]


def test_worker_emits_result(qapp: QApplication) -> None:
    from PyQt6.QtCore import QEventLoop, QTimer

    from fitness_rl.interface.gui.worker import TrainingWorker

    received: list[object] = []
    worker = TrainingWorker(lambda: 42)
    loop = QEventLoop()
    worker.finished_with_result.connect(lambda r: (received.append(r), loop.quit()))
    QTimer.singleShot(2000, loop.quit)
    worker.start()
    loop.exec()
    worker.wait(1000)
    assert received == [42]
