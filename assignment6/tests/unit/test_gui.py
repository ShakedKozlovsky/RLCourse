"""Layer 19 — board renderer + headless GUI core tests."""

from __future__ import annotations

import numpy as np
import pytest

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.game.board import Board
from marl_lab.interface.board_renderer import (
    BARRIER,
    COP,
    EMPTY,
    THIEF,
    ascii_dump,
    render,
)
from marl_lab.interface.game_gui import (
    GameGuiCore,
    make_random_policy,
    make_stay_policy,
)

# ----- Board renderer -----

def _board(cop, thief, barriers=frozenset(), grid=(5, 5)) -> Board:
    return Board(grid_size=grid, cop_pos=cop, thief_pos=thief, barriers=barriers)


def test_render_returns_grid_shape() -> None:
    b = _board(cop=(0, 0), thief=(4, 4))
    g = render(b)
    assert g.shape == (5, 5)
    assert g.dtype == np.int8


def test_render_places_cop_and_thief() -> None:
    b = _board(cop=(2, 3), thief=(1, 4))
    g = render(b)
    assert g[2, 3] == COP
    assert g[1, 4] == THIEF


def test_render_places_barriers() -> None:
    b = _board(cop=(0, 0), thief=(4, 4),
                barriers=frozenset({(1, 1), (2, 2)}))
    g = render(b)
    assert g[1, 1] == BARRIER
    assert g[2, 2] == BARRIER


def test_render_empty_cells_zero() -> None:
    b = _board(cop=(0, 0), thief=(4, 4))
    g = render(b)
    assert g[2, 2] == EMPTY


def test_ascii_dump_produces_visualisation() -> None:
    b = _board(cop=(0, 0), thief=(4, 4))
    text = ascii_dump(b)
    rows = text.split("\n")
    assert len(rows) == 5
    assert rows[0].startswith("C")
    assert rows[-1].endswith("T")


# ----- GUI core -----

@pytest.fixture
def env() -> DecPomdpEnv:
    return DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(4, 4), max_moves=8, max_barriers=2,
                          enable_barriers=False, observation_radius=1),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )


def test_gui_core_reset_initialises_state(env: DecPomdpEnv) -> None:
    gui = GameGuiCore(env=env, cop_policy=make_stay_policy(),
                       thief_policy=make_stay_policy())
    gui.reset(seed=0)
    snap = gui._snapshot()      # noqa: SLF001
    assert snap["steps"] == 0
    assert snap["done"] is False
    assert snap["board_grid"] is not None


def test_gui_core_step_advances_clock(env: DecPomdpEnv) -> None:
    gui = GameGuiCore(env=env, cop_policy=make_stay_policy(),
                       thief_policy=make_stay_policy())
    gui.reset(seed=0)
    snap = gui.step()
    assert snap["steps"] == 1


def test_gui_core_step_before_reset_raises(env: DecPomdpEnv) -> None:
    gui = GameGuiCore(env=env, cop_policy=make_stay_policy(),
                       thief_policy=make_stay_policy())
    with pytest.raises(RuntimeError):
        gui.step()


def test_gui_core_auto_play_terminates(env: DecPomdpEnv) -> None:
    gui = GameGuiCore(env=env, cop_policy=make_stay_policy(),
                       thief_policy=make_stay_policy())
    last = gui.auto_play(max_steps=20)
    # Both agents STAY → no capture → timeout @ max_moves=8 → thief wins
    assert last["done"] is True
    assert last["winner"] == "thief"
    assert last["steps"] == 8


def test_gui_random_policy_is_legal(env: DecPomdpEnv) -> None:
    """Cop random policy should be in [0,6); thief in [0,5)."""
    rng_policy = make_random_policy(rng_seed=0)
    for _ in range(50):
        a_cop = rng_policy("cop", np.zeros(10))
        a_thief = rng_policy("thief", np.zeros(10))
        assert 0 <= a_cop < 6
        assert 0 <= a_thief < 5
