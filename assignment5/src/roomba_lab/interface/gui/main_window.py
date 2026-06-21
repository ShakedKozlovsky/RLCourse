"""PyQt6 main window with Training + Visualisation tabs."""

from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from roomba_lab.interface.gui.training_tab import TrainingTab
from roomba_lab.interface.gui.visualisation_tab import VisualisationTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("roomba-lab — DDPG cleaning robot")
        self.resize(900, 600)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(TrainingTab(), "Training")
        tabs.addTab(VisualisationTab(), "Visualisation")
        self.setCentralWidget(tabs)


def launch() -> None:
    """Spawn the PyQt6 window + start the Qt event loop.

    Blocks until the window is closed. Entry point for `roomba-lab gui`."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
