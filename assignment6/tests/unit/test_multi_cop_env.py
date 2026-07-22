"""Multi-cop env tests (Reflection Q3 support)."""

from __future__ import annotations

import numpy as np
import pytest

from marl_lab.environment.multi_cop_env import MultiCopEnv, MultiCopEnvConfig
from marl_lab.game.actions import Action


def test_multi_cop_env_default_2_cops() -> None:
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=2), rng=np.random.default_rng(0))
    joint_obs = env.reset(seed=0)
    assert "cop_0" in joint_obs
    assert "cop_1" in joint_obs
    assert "thief" in joint_obs


def test_multi_cop_positions_distinct_at_reset() -> None:
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=3, grid_size=(5, 5)),
                        rng=np.random.default_rng(42))
    env.reset(seed=42)
    positions = list(env.board().cop_positions) + [env.board().thief_pos]
    assert len(set(positions)) == 4  # all distinct


def test_multi_cop_step_before_reset_raises() -> None:
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=2))
    with pytest.raises(RuntimeError):
        env.step({"cop_0": 0, "cop_1": 0, "thief": 0})


def test_multi_cop_step_advances_step_counter() -> None:
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=2, max_moves=10))
    env.reset(seed=0)
    env.step({"cop_0": Action.STAY, "cop_1": Action.STAY, "thief": Action.STAY})
    assert env.board().step == 1


def test_multi_cop_capture_ends_episode() -> None:
    """Force a capture by placing cop_0 next to thief and moving in."""
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=2, max_moves=10))
    env.reset(seed=0)
    # Manually set positions: cop_0 at (2, 3), thief at (2, 4)
    from marl_lab.environment.multi_cop_env import MultiCopBoard
    env._board = MultiCopBoard(  # noqa: SLF001
        grid_size=(5, 5),
        cop_positions=[(2, 3), (0, 0)], thief_pos=(2, 4),
    )
    _, r, done, info = env.step({"cop_0": Action.RIGHT,
                                    "cop_1": Action.STAY,
                                    "thief": Action.STAY})
    assert done
    assert info["capture"]
    assert info["winner"] == "cops"
    # Cops share team reward
    assert r["cop_0"] > 0.5
    assert r["cop_1"] > 0.5


def test_multi_cop_timeout_thief_wins() -> None:
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=2, max_moves=3))
    env.reset(seed=0)
    # Everyone stays; no capture happens; must reach timeout at step 3
    for _ in range(3):
        _, _, done, info = env.step({"cop_0": Action.STAY,
                                        "cop_1": Action.STAY,
                                        "thief": Action.STAY})
        if done:
            break
    assert info["winner"] == "thief"


def test_multi_cop_scales_to_n_4() -> None:
    """Q3 study uses N up to 4."""
    env = MultiCopEnv(cfg=MultiCopEnvConfig(n_cops=4, grid_size=(5, 5)))
    env.reset(seed=0)
    joint_obs = env._joint_obs()  # noqa: SLF001
    assert len(joint_obs) == 5  # 4 cops + 1 thief
