"""Shared checkpoint-picker row used by BacktestTab and PredictTab."""

from __future__ import annotations

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class CheckpointPicker(QWidget):
    """A read-write field + a Browse button. Emits no signal — caller reads .path()."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit("")
        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._on_browse)
        h.addWidget(self._edit)
        h.addWidget(self._browse)

    def path(self) -> str:
        """Return the currently entered checkpoint file path."""
        return self._edit.text()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pick checkpoint", filter="*.pt")
        if path:
            self._edit.setText(path)
