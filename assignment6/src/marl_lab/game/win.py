"""Win adjudication — spec § 3.2.

Pure function on (Board, max_moves). Returns winner or None (sub-game ongoing)."""

from __future__ import annotations

from typing import Literal

from marl_lab.game.board import Board

Winner = Literal["cop", "thief", None]


def adjudicate(board: Board, max_moves: int) -> Winner:
    """Return 'cop' / 'thief' / None per spec § 3.2.

    cop wins  ⇔  cop position == thief position (capture)
    thief wins ⇔ step >= max_moves AND no capture this turn"""
    if board.capture_flag:
        return "cop"
    if board.step >= int(max_moves):
        return "thief"
    return None
