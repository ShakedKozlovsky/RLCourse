"""Render each GUI tab to a PNG under assets/gui/.

Runs under QT_QPA_PLATFORM=offscreen so it works without a display server.
Usage: uv run python scripts/capture_gui_screenshots.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from dqn_trader.interface.gui.main_window import MainWindow  # noqa: E402
from dqn_trader.sdk.sdk import TradingSDK  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "gui"
OUT.mkdir(parents=True, exist_ok=True)


def _save(widget, name: str) -> None:
    widget.repaint()
    QApplication.processEvents()
    pixmap = widget.grab()
    target = OUT / f"{name}.png"
    pixmap.save(str(target), "PNG")
    print(f"wrote {target}")


def main() -> int:
    _ = QApplication(sys.argv)
    sdk = TradingSDK()
    win = MainWindow(sdk)
    win.resize(1024, 720)
    win.show()
    QApplication.processEvents()

    for idx, name in enumerate(("data", "train", "backtest", "predict")):
        win.tabs.setCurrentIndex(idx)
        QApplication.processEvents()
        _save(win, f"tab_{name}")

    # Bonus: render the data tab *after* clicking Prepare to show real shapes.
    win.tabs.setCurrentIndex(0)
    win.data_tab._on_prepare()
    QApplication.processEvents()
    _save(win, "tab_data_after_prepare")

    win.close()
    _ = Qt  # silence unused import
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
