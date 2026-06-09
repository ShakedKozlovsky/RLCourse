"""QThread worker — runs blocking SDK calls off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """One-shot QThread that runs ``fn()`` and emits the result via signals."""

    finished_with_result = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, fn: Callable[[], Any], parent: object | None = None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(exc)
            return
        self.finished_with_result.emit(result)
