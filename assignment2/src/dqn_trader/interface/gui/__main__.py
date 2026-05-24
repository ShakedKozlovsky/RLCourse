"""Entry point: ``python -m dqn_trader.interface.gui``."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from dqn_trader.interface.gui.main_window import MainWindow
from dqn_trader.sdk.sdk import TradingSDK


def main() -> int:
    """Launch the PyQt6 GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow(TradingSDK())
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
