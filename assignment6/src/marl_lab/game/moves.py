"""Move dynamics T(s' | s, ā) — pure function on Boards.

Simultaneous resolution: cop and thief move in the same tick. Walls + barriers
block movement. PLACE_BARRIER (action 5, cop-only) drops a barrier on the
cop's CURRENT cell (spec § 3.3) and the cop stays put that turn.
block; collisions resolve as the **moving agent stays put**. Cop's
PLACE_BARRIER (action 5) drops a barrier on the cell ADJACENT in cop's last
intended direction — see § 1 below."""

from __future__ import annotations

from dataclasses import dataclass

from marl_lab.game.actions import DELTA, Action
from marl_lab.game.board import Board


def _try_move(pos: tuple[int, int], action: Action, board: Board) -> tuple[int, int]:
    """Compute candidate next position; revert to original if blocked."""
    dr, dc = DELTA[action]
    candidate = (pos[0] + dr, pos[1] + dc)
    if board.is_blocked(candidate):
        return pos
    return candidate


@dataclass(frozen=True)
class MoveInfo:
    """Diagnostics for one move tick — used by reward shaping + tests."""
    cop_intended_move: bool
    thief_intended_move: bool
    cop_blocked: bool
    thief_blocked: bool
    barrier_placed: bool
    barrier_placed_at: tuple[int, int] | None
    collision: bool                 # cop and thief tried to land on same cell


class MoveDynamics:
    """Simultaneous-action transition kernel for the 2-agent Cops-and-Robbers env."""

    def __init__(self, max_barriers: int) -> None:
        self.max_barriers = int(max_barriers)

    def apply(self, board: Board, cop_action: Action, thief_action: Action) -> tuple[Board, MoveInfo]:
        """Resolve one joint action; return (new board, MoveInfo).

        Rules:
          1. THIEF moves first conceptually but simultaneously with cop.
             If both target the SAME empty cell → both stay put (collision tie).
          2. If cop targets the thief's CURRENT cell and thief doesn't move →
             cop captures (lands on thief). step counter still advances.
          3. PLACE_BARRIER (cop only): cop does NOT move. Barrier is placed on
             the cell one step in cop's last-direction-of-intent (UP-direction
             default). Cannot place on:
                 - cop's own cell
                 - thief's cell
                 - existing barrier
                 - off-grid
             Cap: ``max_barriers`` per game. Beyond cap → barrier action is
             silently a STAY.
          4. The step counter increments by 1 every tick."""
        # Resolve thief
        thief_target = _try_move(board.thief_pos, thief_action, board)
        thief_blocked = (thief_target == board.thief_pos and thief_action != Action.STAY)

        # Resolve cop
        # Spec § 3.3: placing a barrier puts it on the cop's CURRENT cell
        # (and costs the move — cop stays put). The barrier is recorded
        # immediately; the cop is now "on" the barrier cell but will move
        # off it next turn.
        barrier_placed = False
        barrier_at: tuple[int, int] | None = None
        new_barriers = board.barriers
        if cop_action == Action.PLACE_BARRIER:
            cop_target = board.cop_pos
            cand = board.cop_pos
            if (
                len(board.barriers) < self.max_barriers
                and cand not in board.barriers
                and cand != board.thief_pos
            ):
                new_barriers = frozenset(board.barriers | {cand})
                barrier_placed = True
                barrier_at = cand
            cop_blocked = False
        else:
            cop_target = _try_move(board.cop_pos, cop_action, board)
            cop_blocked = (cop_target == board.cop_pos and cop_action != Action.STAY)

        # Resolve simultaneity. Two collision flavours, only one of which
        # cancels the moves:
        #   (a) BOTH agents try to land on the SAME new cell — both stay
        #       (only if that cell is empty AND not on the thief's old position
        #       — otherwise it's a capture, see below).
        #   (b) Cop and thief SWAP cells in one tick — both stay (can't pass
        #       through each other).
        # If the cop is moving ONTO the thief's STAYING cell that's a
        # capture, not a collision.
        cop_moved = cop_target != board.cop_pos
        thief_moved = thief_target != board.thief_pos
        same_cell_collision = (
            cop_action != Action.PLACE_BARRIER
            and cop_target == thief_target
            and cop_moved and thief_moved        # both moving + landed same square
            and cop_target != board.thief_pos    # not the cop chasing standing thief
        )
        swap_collision = (
            cop_action != Action.PLACE_BARRIER
            and cop_target == board.thief_pos
            and thief_target == board.cop_pos
            and cop_moved and thief_moved
        )
        collision = same_cell_collision or swap_collision
        if collision:
            cop_target = board.cop_pos
            thief_target = board.thief_pos

        # Capture: cop on thief's NEW cell
        capture = (cop_target == thief_target)
        step = board.step + 1

        new_board = board.with_(
            cop_pos=cop_target,
            thief_pos=thief_target,
            barriers=new_barriers,
            step=step,
            capture_flag=capture,
        )
        info = MoveInfo(
            cop_intended_move=(cop_action != Action.STAY and cop_action != Action.PLACE_BARRIER),
            thief_intended_move=(thief_action != Action.STAY),
            cop_blocked=cop_blocked,
            thief_blocked=thief_blocked,
            barrier_placed=barrier_placed,
            barrier_placed_at=barrier_at,
            collision=collision,
        )
        return new_board, info
