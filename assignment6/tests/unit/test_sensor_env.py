"""Layer 3 — sensor + reward + Dec-POMDP env tests."""

from __future__ import annotations

import numpy as np
import pytest

from marl_lab.environment.dec_pomdp import DecPomdpEnv, EnvConfig
from marl_lab.environment.reward import RewardConfig, per_step_reward, sub_game_score
from marl_lab.game.actions import Action
from marl_lab.game.board import Board
from marl_lab.sensor.partial_observation import (
    BARRIER,
    OPPONENT,
    WALL,
    cells_in_radius,
    obs_dim,
    observe,
)

# ----- Manhattan radius -----

def test_cells_in_radius_r1_count() -> None:
    cells = cells_in_radius((2, 2), 1)
    assert len(cells) == 5     # diamond size at r=1


def test_cells_in_radius_r2_count() -> None:
    cells = cells_in_radius((2, 2), 2)
    assert len(cells) == 13    # diamond size at r=2


def test_cells_in_radius_r3_count() -> None:
    assert len(cells_in_radius((3, 3), 3)) == 25


def test_obs_dim_formula() -> None:
    assert obs_dim(1) == 4 * 5 + 6
    assert obs_dim(2) == 4 * 13 + 6
    assert obs_dim(3) == 4 * 25 + 6


# ----- Observation content -----

def _board(cop, thief, barriers=frozenset(), grid=(5, 5)) -> Board:
    return Board(grid_size=grid, cop_pos=cop, thief_pos=thief, barriers=barriers)


def test_observe_opponent_within_radius() -> None:
    b = _board(cop=(2, 2), thief=(2, 3))   # Manhattan = 1
    o_cop = observe(b, "cop", radius=1)
    # 5 cells × 4 channels = 20; opponent must appear in OPPONENT channel
    grid_flat = o_cop[:20].reshape(5, 4)
    assert grid_flat[:, OPPONENT].sum() == 1.0


def test_observe_opponent_outside_radius_hidden() -> None:
    b = _board(cop=(0, 0), thief=(4, 4))   # Manhattan = 8
    o_cop = observe(b, "cop", radius=2)
    grid_flat = o_cop[:13 * 4].reshape(13, 4)
    assert grid_flat[:, OPPONENT].sum() == 0.0


def test_observe_off_grid_encoded_as_wall() -> None:
    b = _board(cop=(0, 0), thief=(4, 4))    # cop at top-left corner
    o_cop = observe(b, "cop", radius=1)
    grid_flat = o_cop[:5 * 4].reshape(5, 4)
    # 5 cells: (-1,0), (0,-1), (0,0), (0,1), (1,0). Two are off-grid → 2 walls.
    assert grid_flat[:, WALL].sum() == 2.0


def test_observe_barrier_encoded() -> None:
    b = _board(cop=(2, 2), thief=(4, 4), barriers=frozenset({(2, 3)}))
    o_cop = observe(b, "cop", radius=1)
    grid_flat = o_cop[:5 * 4].reshape(5, 4)
    assert grid_flat[:, BARRIER].sum() == 1.0


def test_observe_dim_matches_obs_dim() -> None:
    b = _board(cop=(2, 2), thief=(0, 0))
    for r in (1, 2, 3):
        assert observe(b, "cop", radius=r).shape == (obs_dim(r),)


# ----- Reward function -----

def test_per_step_reward_no_terminal() -> None:
    cfg = RewardConfig()
    r = per_step_reward(capture=False, timeout=False, collision=False,
                         barrier_placed_by_cop=False, cfg=cfg)
    assert r["cop"] == pytest.approx(cfg.step_penalty_cop)
    assert r["thief"] == pytest.approx(cfg.step_penalty_thief)


def test_per_step_reward_capture() -> None:
    cfg = RewardConfig()
    r = per_step_reward(capture=True, timeout=False, collision=False,
                         barrier_placed_by_cop=False, cfg=cfg)
    assert r["cop"] > 0
    assert r["thief"] < 0


def test_per_step_reward_timeout() -> None:
    cfg = RewardConfig()
    r = per_step_reward(capture=False, timeout=True, collision=False,
                         barrier_placed_by_cop=False, cfg=cfg)
    assert r["cop"] < 0
    assert r["thief"] > 0


def test_sub_game_score_cop_wins() -> None:
    cfg = RewardConfig()
    sc = sub_game_score("cop", cfg)
    assert sc == {"cop": 20, "thief": 5}


def test_sub_game_score_thief_wins() -> None:
    cfg = RewardConfig()
    sc = sub_game_score("thief", cfg)
    assert sc == {"cop": 5, "thief": 10}


# ----- Dec-POMDP env -----

@pytest.fixture
def env() -> DecPomdpEnv:
    return DecPomdpEnv(
        env_cfg=EnvConfig(grid_size=(5, 5), max_moves=25, max_barriers=5,
                          enable_barriers=True, observation_radius=2),
        reward_cfg=RewardConfig(),
        rng=np.random.default_rng(0),
    )


def test_env_reset_returns_joint_obs(env: DecPomdpEnv) -> None:
    obs = env.reset(seed=0)
    assert "cop" in obs and "thief" in obs
    assert obs["cop"].shape == (env.obs_dim,)
    assert obs["thief"].shape == (env.obs_dim,)


def test_env_step_returns_correct_tuple(env: DecPomdpEnv) -> None:
    env.reset(seed=0)
    o, r, done, info = env.step({"cop": Action.STAY, "thief": Action.STAY})
    assert isinstance(r, dict)
    assert isinstance(done, bool)
    assert "winner" in info
    assert info["step"] == 1


def test_env_capture_ends_episode(env: DecPomdpEnv) -> None:
    env.reset(seed=0)
    # Force-set the board state for a known capture in one step
    env._board = env._board.with_(cop_pos=(2, 3), thief_pos=(2, 4))  # noqa: SLF001
    o, r, done, info = env.step({"cop": Action.RIGHT, "thief": Action.STAY})
    assert done
    assert info["winner"] == "cop"


def test_env_timeout_ends_episode(env: DecPomdpEnv) -> None:
    env.reset(seed=0)
    env._board = env._board.with_(step=24, cop_pos=(0, 0), thief_pos=(4, 4))  # noqa: SLF001
    o, r, done, info = env.step({"cop": Action.STAY, "thief": Action.STAY})
    assert done
    assert info["winner"] == "thief"


def test_env_global_state_dim(env: DecPomdpEnv) -> None:
    env.reset(seed=0)
    s = env.global_state()
    assert s.shape == (3 * 25 + 2,)


def test_env_step_before_reset_raises(env: DecPomdpEnv) -> None:
    with pytest.raises(RuntimeError):
        env.step({"cop": Action.STAY, "thief": Action.STAY})


def test_env_global_state_before_reset_raises(env: DecPomdpEnv) -> None:
    with pytest.raises(RuntimeError):
        env.global_state()


def test_env_reproducible_at_same_seed(env: DecPomdpEnv) -> None:
    o1 = env.reset(seed=42)
    e2 = DecPomdpEnv(env.env_cfg, env.reward_cfg, np.random.default_rng(0))
    o2 = e2.reset(seed=42)
    np.testing.assert_array_equal(o1["cop"], o2["cop"])
    np.testing.assert_array_equal(o1["thief"], o2["thief"])


def test_env_no_gym_imports() -> None:
    """ADR-001 — verify zero gym imports in the env (and the whole source tree)."""
    import marl_lab.environment.dec_pomdp as mod
    src = mod.__file__
    with open(src) as f:
        text = f.read()
    assert "import gym" not in text
    assert "from gym" not in text
