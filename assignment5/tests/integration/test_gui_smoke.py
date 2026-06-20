"""Layer 14 — PyQt6 GUI smoke under offscreen Qt."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


def test_main_window_constructs() -> None:
    from PyQt6 import QtWidgets

    from roomba_lab.interface.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    assert win.windowTitle().startswith("roomba-lab")
    win.close()
    _ = app


def test_training_tab_constructs() -> None:
    from PyQt6 import QtWidgets

    from roomba_lab.interface.gui.training_tab import TrainingTab
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = TrainingTab()
    assert w.steps_box.value() >= 100


def test_visualisation_tab_constructs() -> None:
    from PyQt6 import QtWidgets

    from roomba_lab.interface.gui.visualisation_tab import VisualisationTab
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = VisualisationTab()
    assert w.checkpoint_path.text().endswith(".pt")
