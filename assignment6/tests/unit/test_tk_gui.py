"""Tk GUI tests — module imports cleanly on headless envs; launch fails cleanly."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def test_tk_gui_module_imports_on_headless() -> None:
    """`import interface.tk_gui` must not crash even with no DISPLAY."""
    import marl_lab.interface.tk_gui as mod
    # Must expose the public API
    assert hasattr(mod, "LiveTkGui")
    assert hasattr(mod, "launch_live_gui")


def test_launch_live_gui_reports_no_display() -> None:
    """On headless (no DISPLAY), launch_live_gui raises SystemExit with a
    helpful message pointing at the pre-rendered GIF alternative."""
    from marl_lab.interface.tk_gui import launch_live_gui
    with patch.dict(os.environ, {}, clear=True), pytest.raises(SystemExit) as e:
        launch_live_gui(checkpoint_path="/nonexistent.pt")
    assert "DISPLAY" in str(e.value) or "sub_game.gif" in str(e.value)


def test_launch_live_gui_reports_missing_checkpoint() -> None:
    """On a display-capable env, missing checkpoint gives a clear error."""
    from marl_lab.interface.tk_gui import launch_live_gui
    with patch.dict(os.environ, {"DISPLAY": ":0"}), pytest.raises(SystemExit) as e:
        launch_live_gui(checkpoint_path="/definitely/not/here.pt")
    assert "checkpoint not found" in str(e.value)
