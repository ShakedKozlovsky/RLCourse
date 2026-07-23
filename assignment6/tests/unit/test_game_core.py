"""Layer 2 — game core tests (board + moves + win adjudication + barriers)."""

from __future__ import annotations

import numpy as np

from marl_lab.game.actions import Action, n_actions
from marl_lab.game.board import Board, BoardFactory
from marl_lab.game.moves import MoveDynamics
from marl_lab.game.win import adjudicate

# ----- Action set -----

def test_n_actions_cop_with_barriers() -> None:
    assert n_actions("cop", enable_barriers=True) == 6


def test_n_actions_cop_no_barriers() -> None:
    assert n_actions("cop", enable_barriers=False) == 5


def test_n_actions_thief() -> None:
    assert n_actions("thief", enable_barriers=True) == 5


# ----- Board -----

def test_board_in_bounds() -> None:
    b = Board(grid_size=(5, 5), cop_pos=(0, 0), thief_pos=(4, 4))
    assert b.in_bounds((2, 2))
    assert not b.in_bounds((-1, 0))
    assert not b.in_bounds((5, 5))


def test_board_cell_kind() -> None:
    b = Board(grid_size=(5, 5), cop_pos=(0, 0), thief_pos=(4, 4),
              barriers=frozenset({(2, 2)}))
    assert b.cell_kind((0, 0)) == "cop"
    assert b.cell_kind((4, 4)) == "thief"
    assert b.cell_kind((2, 2)) == "barrier"
    assert b.cell_kind((1, 1)) == "empty"
    assert b.cell_kind((-1, 0)) == "offgrid"


def test_board_state_vector_dim() -> None:
    b = Board(grid_size=(5, 5), cop_pos=(0, 0), thief_pos=(4, 4))
    v = b.to_state_vector()
    assert v.shape == (3 * 25 + 2,)        # 77
    assert Board.state_vector_dim((5, 5)) == 77


def test_board_state_vector_encoding() -> None:
    b = Board(grid_size=(3, 3), cop_pos=(0, 0), thief_pos=(2, 2),
              barriers=frozenset({(1, 1)}), step=5)
    v = b.to_state_vector()
    # cop at index 0 (row 0 col 0); thief at index 8 (row 2 col 2)
    assert v[0] == 1.0
    assert v[9 + 8] == 1.0                  # thief block starts at index 9
    assert v[18 + 4] == 1.0                 # barrier at (1,1) → idx 4 in barrier block
    assert v[-2] == 5.0                     # step scalar
    assert v[-1] == 1.0                     # barrier count


def test_board_factory_distinct_positions() -> None:
    bf = BoardFactory(grid_size=(5, 5), rng=np.random.default_rng(0))
    for _ in range(20):
        b = bf.fresh()
        assert b.cop_pos != b.thief_pos


# ----- MoveDynamics -----

def _make_board(cop, thief, barriers=frozenset(), grid=(5, 5)) -> Board:
    return Board(grid_size=grid, cop_pos=cop, thief_pos=thief, barriers=barriers)


def test_move_basic_displacement() -> None:
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(2, 2), thief=(0, 0))
    new_b, _ = md.apply(b, Action.RIGHT, Action.DOWN)
    assert new_b.cop_pos == (2, 3)
    assert new_b.thief_pos == (1, 0)
    assert new_b.step == 1


def test_move_walls_block() -> None:
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(0, 0), thief=(4, 4))
    new_b, info = md.apply(b, Action.UP, Action.RIGHT)  # cop UP from row 0 is off-grid
    assert new_b.cop_pos == (0, 0)
    assert info.cop_blocked


def test_move_barriers_block() -> None:
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(2, 2), thief=(4, 4), barriers=frozenset({(2, 3)}))
    new_b, info = md.apply(b, Action.RIGHT, Action.STAY)
    assert new_b.cop_pos == (2, 2)
    assert info.cop_blocked


def test_move_capture_on_overlap() -> None:
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(2, 3), thief=(2, 4))
    new_b, _ = md.apply(b, Action.RIGHT, Action.STAY)
    assert new_b.cop_pos == (2, 4)
    assert new_b.capture_flag


def test_move_simultaneous_target_collision() -> None:
    """If both target the same empty cell, both stay."""
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(1, 1), thief=(1, 3))
    new_b, info = md.apply(b, Action.RIGHT, Action.LEFT)
    # both targeting (1, 2)
    assert new_b.cop_pos == (1, 1)
    assert new_b.thief_pos == (1, 3)
    assert info.collision


def test_barrier_placement_basic() -> None:
    """Spec § 3.3 — barrier goes on the COP'S OWN cell, cop stays put."""
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(2, 2), thief=(4, 4))
    new_b, info = md.apply(b, Action.PLACE_BARRIER, Action.STAY)
    assert info.barrier_placed
    assert info.barrier_placed_at == (2, 2)
    assert (2, 2) in new_b.barriers
    assert len(new_b.barriers) == 1
    assert new_b.cop_pos == (2, 2)


def test_barrier_cap_enforced() -> None:
    md = MoveDynamics(max_barriers=2)
    b = _make_board(cop=(3, 3), thief=(4, 4),
                    barriers=frozenset({(0, 0), (1, 1)}))
    new_b, info = md.apply(b, Action.PLACE_BARRIER, Action.STAY)
    assert not info.barrier_placed
    assert len(new_b.barriers) == 2


def test_barrier_cannot_place_when_thief_on_same_cell() -> None:
    """Edge case: cop and thief on the same cell mid-game (pre-capture). Spec
    forbids placing the barrier under the thief."""
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(2, 2), thief=(2, 2))  # same cell
    new_b, info = md.apply(b, Action.PLACE_BARRIER, Action.STAY)
    assert not info.barrier_placed
    assert (2, 2) not in new_b.barriers


def test_barrier_cannot_double_place_same_cell() -> None:
    """If cop's cell already has a barrier, no-op (per spec § 3.3)."""
    md = MoveDynamics(max_barriers=5)
    b = _make_board(cop=(0, 2), thief=(4, 4), barriers=frozenset({(0, 2)}))
    new_b, info = md.apply(b, Action.PLACE_BARRIER, Action.STAY)
    assert not info.barrier_placed
    assert len(new_b.barriers) == 1


# ----- Win adjudication -----

def test_adjudicate_capture_returns_cop() -> None:
    b = _make_board(cop=(2, 4), thief=(2, 4)).with_(capture_flag=True)
    assert adjudicate(b, max_moves=25) == "cop"


def test_adjudicate_timeout_returns_thief() -> None:
    b = _make_board(cop=(0, 0), thief=(4, 4)).with_(step=25)
    assert adjudicate(b, max_moves=25) == "thief"


def test_adjudicate_ongoing() -> None:
    b = _make_board(cop=(0, 0), thief=(4, 4)).with_(step=10)
    assert adjudicate(b, max_moves=25) is None
