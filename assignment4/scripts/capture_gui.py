"""Capture one PNG per GUI tab — headless via QT_QPA_PLATFORM=offscreen."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from proximal_lab.interface.gui.main_window import MainWindow  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "gui"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    cfg = Path(__file__).resolve().parents[1] / "configs" / "setup.json"
    window = MainWindow(config_path=cfg)
    window.resize(QSize(1100, 700))
    window.show()
    app.processEvents()
    # Trigger the sweep + compare tabs to render their plots
    sweep_tab = window.tabs.widget(1)
    sweep_tab._on_plot()  # noqa: SLF001
    app.processEvents()
    compare_tab = window.tabs.widget(2)
    compare_tab._on_plot()  # noqa: SLF001
    app.processEvents()
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        app.processEvents()
        name = window.tabs.tabText(i).replace(" ", "_").lower()
        pixmap = window.grab()
        pixmap.save(str(OUT / f"tab_{i}_{name}.png"), "PNG")
        print(f"wrote {OUT}/tab_{i}_{name}.png")
    app.quit()


if __name__ == "__main__":
    main()
