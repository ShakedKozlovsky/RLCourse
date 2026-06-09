"""proximal-lab GUI — 3 tabs (Train · Sweep · Compare) on top of the SDK."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget

from proximal_lab.interface.gui.compare_tab import CompareTab
from proximal_lab.interface.gui.sweep_tab import SweepTab
from proximal_lab.interface.gui.train_tab import TrainTab
from proximal_lab.sdk.sdk import ProximalLab


class MainWindow(QMainWindow):
    """QMainWindow holding the shared SDK and 3 tabs."""

    def __init__(self, config_path: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("proximal-lab — PPO + GAE on MuJoCo")
        self.resize(900, 650)
        self._sdk = ProximalLab(config_path=config_path)
        self._tabs = QTabWidget(self)
        self._tabs.addTab(TrainTab(self._sdk), "Train")
        self._tabs.addTab(SweepTab(self._sdk), "Sweep")
        self._tabs.addTab(CompareTab(self._sdk), "Compare")
        self.setCentralWidget(self._tabs)

    @property
    def sdk(self) -> ProximalLab:
        return self._sdk

    @property
    def tabs(self) -> QTabWidget:
        return self._tabs


def launch(config_path: Path) -> int:  # pragma: no cover
    import sys

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(config_path=config_path)
    window.show()
    return app.exec()
