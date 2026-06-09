"""Matplotlib FigureCanvas wrapped in a QWidget — used by every GUI tab."""

from __future__ import annotations

from collections.abc import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class PlotWidget(QWidget):
    """Single-axes plot; ``draw(fn)`` clears and re-renders."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def draw(self, fn: Callable[[object], None]) -> None:
        self._ax.clear()
        fn(self._ax)
        self._canvas.draw_idle()
