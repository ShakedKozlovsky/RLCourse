"""Pure board-to-grid renderer — testable without Tkinter.

Returns a (h, w) numpy int8 array where each cell is one of:
  0 EMPTY, 1 COP, 2 THIEF, 3 BARRIER. The Tkinter layer reads this
and paints cells; the notebook can render it directly via matplotlib."""

from __future__ import annotations

import numpy as np

from marl_lab.game.board import Board

EMPTY, COP, THIEF, BARRIER = 0, 1, 2, 3


def render(board: Board) -> np.ndarray:
    """Return an int8 (h, w) grid encoding cell contents."""
    h, w = board.grid_size
    grid = np.zeros((h, w), dtype=np.int8)
    for r, c in board.barriers:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = BARRIER
    cr, cc = board.cop_pos
    tr, tc = board.thief_pos
    grid[cr, cc] = COP
    grid[tr, tc] = THIEF
    return grid


def ascii_dump(board: Board) -> str:
    """ASCII visualisation — debugging aid (also used by GUI status line)."""
    grid = render(board)
    chars = {EMPTY: ".", COP: "C", THIEF: "T", BARRIER: "#"}
    rows = ["".join(chars[v] for v in row) for row in grid]
    return "\n".join(rows)
