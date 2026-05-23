"""Top-level GUI window that hosts the four tabs."""

from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from dqn_trader.interface.gui.backtest_tab import BacktestTab
from dqn_trader.interface.gui.data_tab import DataTab
from dqn_trader.interface.gui.predict_tab import PredictTab
from dqn_trader.interface.gui.train_tab import TrainTab
from dqn_trader.sdk.sdk import TradingSDK
from dqn_trader.shared.version import __version__


class MainWindow(QMainWindow):
    """Hosts a QTabWidget. The window itself is a thin shell — tabs do the work."""

    def __init__(self, sdk: TradingSDK) -> None:
        super().__init__()
        self.setWindowTitle(f"DQN Trader — v{__version__}")
        self.resize(1024, 720)
        self.tabs = QTabWidget()
        self.data_tab = DataTab(sdk)
        self.train_tab = TrainTab(sdk)
        self.backtest_tab = BacktestTab(sdk)
        self.predict_tab = PredictTab(sdk)
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.backtest_tab, "Backtest")
        self.tabs.addTab(self.predict_tab, "Predict")
        self.setCentralWidget(self.tabs)
