"""Main GUI window — 5 tabs sitting on top of the FitnessRL SDK."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget

from fitness_rl.interface.gui.algo_tab import A2CTab, ReinforceTab
from fitness_rl.interface.gui.compare_tab import CompareTab
from fitness_rl.interface.gui.data_tab import DataTab
from fitness_rl.interface.gui.world_model_tab import WorldModelTab
from fitness_rl.sdk.sdk import FitnessRL


class MainWindow(QMainWindow):
    """QMainWindow holding the shared SDK and the 5 functional tabs."""

    def __init__(self, config_path: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("fitness-rl — REINFORCE + A2C")
        self.resize(900, 700)
        self._sdk = FitnessRL(config_path=config_path)
        tabs = QTabWidget(self)
        tabs.addTab(DataTab(self._sdk), "Data")
        tabs.addTab(WorldModelTab(self._sdk), "World model")
        tabs.addTab(ReinforceTab(self._sdk), "REINFORCE")
        tabs.addTab(A2CTab(self._sdk), "A2C")
        tabs.addTab(CompareTab(self._sdk), "Compare")
        self.setCentralWidget(tabs)
        self._tabs = tabs

    @property
    def sdk(self) -> FitnessRL:
        return self._sdk

    @property
    def tabs(self) -> QTabWidget:
        return self._tabs


def launch(config_path: Path) -> int:  # pragma: no cover - real GUI entry-point
    """Build the QApplication and show the main window."""
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(config_path=config_path)
    window.show()
    return app.exec()
