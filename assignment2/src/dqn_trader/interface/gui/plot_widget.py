"""Matplotlib canvas embedded into a Qt widget.

Why we don't lazily import here: this module is only imported by the GUI
package, which already pulls in PyQt6. Same for matplotlib.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


class PlotWidget(FigureCanvas):
    """A FigureCanvas with two helper methods for the curves the GUI shows."""

    def __init__(self, width: float = 5.0, height: float = 3.0, dpi: int = 100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self._ax = fig.add_subplot(111)
        super().__init__(fig)

    def plot_equity(self, equity: np.ndarray, benchmark: np.ndarray | None = None) -> None:
        self._ax.clear()
        self._ax.plot(equity, label="DQN policy", linewidth=2)
        if benchmark is not None:
            self._ax.plot(benchmark, label="Buy & Hold", linestyle="--")
        self._ax.set_xlabel("step")
        self._ax.set_ylabel("portfolio value")
        self._ax.legend()
        self._ax.grid(alpha=0.3)
        self.draw()

    def plot_metric(self, values: list[float], title: str) -> None:
        self._ax.clear()
        self._ax.plot(values, marker="o")
        self._ax.set_title(title)
        self._ax.set_xlabel("episode")
        self._ax.grid(alpha=0.3)
        self.draw()
