"""Property-based fuzz tests for the env + observation + win adjudicator.

Hypothesis generates random valid (board, action) inputs and asserts
invariants that MUST hold regardless of the specific values. This is
stronger than example-based testing: it explores corner-cases we'd never
think to write down.

Invariants under test:
  1. cop_pos and thief_pos always stay in-bounds after any move
  2. barrier count NEVER exceeds max_barriers
  3. observation vector dimension is exactly obs_dim(radius)
  4. observation never reveals opponent if Manhattan distance > radius
  5. capture detected ⇒ cop_pos == thief_pos at the moment of capture
  6. step counter monotonically increases by exactly 1 per .apply()
  7. barriers are always a subset of the grid cells
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig
from marl_lab.game.actions import Action
from marl_lab.game.board import Board
from marl_lab.game.moves import MoveDynamics
from marl_lab.sensor.partial_observation import OPPONENT, obs_dim, observe

# ---- Strategies ----

@st.composite
def grid_size_strategy(draw):
    n = draw(st.integers(min_value=2, max_value=6))
    return (n, n)


@st.composite
def board_strategy(draw):
    """Generate a valid Board with cop/thief at distinct in-bounds cells."""
    grid = draw(grid_size_strategy())
    h, w = grid
    cells = [(r, c) for r in range(h) for c in range(w)]
    cop = draw(st.sampled_from(cells))
    thief_candidates = [c for c in cells if c != cop]
    thief = draw(st.sampled_from(thief_candidates))
    # Up to 3 random barriers, distinct from cop+thief
    n_barriers = draw(st.integers(min_value=0, max_value=min(3, len(cells) - 2)))
    barrier_pool = [c for c in cells if c != cop and c != thief]
    barriers = draw(st.lists(st.sampled_from(barrier_pool),
                              min_size=n_barriers, max_size=n_barriers,
                              unique=True)) if n_barriers > 0 else []
    return Board(grid_size=grid, cop_pos=cop, thief_pos=thief,
                  barriers=frozenset(barriers))


cop_action_strategy = st.sampled_from([Action.UP, Action.DOWN, Action.LEFT,
                                          Action.RIGHT, Action.STAY,
                                          Action.PLACE_BARRIER])
thief_action_strategy = st.sampled_from([Action.UP, Action.DOWN, Action.LEFT,
                                            Action.RIGHT, Action.STAY])


# ---- Invariants ----

@given(board=board_strategy(), cop_a=cop_action_strategy, thief_a=thief_action_strategy)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_positions_stay_in_bounds(board: Board, cop_a: Action, thief_a: Action) -> None:
    """INV 1: after .apply(), cop_pos and thief_pos are always in-bounds."""
    md = MoveDynamics(max_barriers=5)
    new_board, _ = md.apply(board, cop_a, thief_a)
    assert new_board.in_bounds(new_board.cop_pos)
    assert new_board.in_bounds(new_board.thief_pos)


@given(board=board_strategy(), cop_a=cop_action_strategy, thief_a=thief_action_strategy)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_barrier_count_never_exceeds_cap(board: Board, cop_a: Action,
                                           thief_a: Action) -> None:
    """INV 2: |new_barriers| ≤ max_barriers for every action sequence."""
    md = MoveDynamics(max_barriers=3)
    new_board, _ = md.apply(board, cop_a, thief_a)
    assert len(new_board.barriers) <= 3


@given(board=board_strategy(), cop_a=cop_action_strategy, thief_a=thief_action_strategy)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_step_counter_increments_by_one(board: Board, cop_a: Action,
                                          thief_a: Action) -> None:
    """INV 6: each .apply() advances step by exactly 1."""
    md = MoveDynamics(max_barriers=5)
    new_board, _ = md.apply(board, cop_a, thief_a)
    assert new_board.step == board.step + 1


@given(board=board_strategy(), cop_a=cop_action_strategy, thief_a=thief_action_strategy)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_barriers_always_in_grid(board: Board, cop_a: Action, thief_a: Action) -> None:
    """INV 7: every barrier coord is in [0, h) × [0, w)."""
    md = MoveDynamics(max_barriers=5)
    new_board, _ = md.apply(board, cop_a, thief_a)
    h, w = new_board.grid_size
    for r, c in new_board.barriers:
        assert 0 <= r < h
        assert 0 <= c < w


@given(board=board_strategy(),
       radius=st.integers(min_value=1, max_value=3),
       role=st.sampled_from(["cop", "thief"]))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_observation_dim_matches_formula(board: Board, radius: int,
                                            role: str) -> None:
    """INV 3: observe(...).shape == (obs_dim(radius),)."""
    o = observe(board, role, radius=radius)
    assert o.shape == (obs_dim(radius),)


@given(board=board_strategy(),
       radius=st.integers(min_value=1, max_value=3),
       role=st.sampled_from(["cop", "thief"]))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_opponent_hidden_outside_radius(board: Board, radius: int,
                                           role: str) -> None:
    """INV 4: if Manhattan(self, opp) > radius, the OPPONENT channel is empty."""
    self_pos = board.cop_pos if role == "cop" else board.thief_pos
    opp_pos = board.thief_pos if role == "cop" else board.cop_pos
    manhattan = abs(self_pos[0] - opp_pos[0]) + abs(self_pos[1] - opp_pos[1])
    o = observe(board, role, radius=radius)
    n_visible = 2 * radius * (radius + 1) + 1
    grid_flat = o[:n_visible * 4].reshape(n_visible, 4)
    if manhattan > radius:
        assert grid_flat[:, OPPONENT].sum() == 0.0
    else:
        # Opponent must be visible exactly once
        assert grid_flat[:, OPPONENT].sum() == 1.0


# ---- Full-env invariant (capture detection) ----

@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_capture_implies_positions_equal(seed: int) -> None:
    """INV 5: when env.step() reports winner='cop', positions must coincide.

    We force-run random rollouts and check the capture invariant whenever
    it fires."""
    env = DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(4, 4), max_moves=15, max_barriers=2,
                          enable_barriers=False, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(seed),
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(15):
        a = {"cop": int(rng.integers(0, 6)), "thief": int(rng.integers(0, 5))}
        _, _, done, info = env.step(a)
        if done and info["winner"] == "cop":
            board = env.board()
            assert board.cop_pos == board.thief_pos
        if done:
            break
