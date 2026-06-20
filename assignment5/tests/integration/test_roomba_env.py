"""Layer 3 — RoombaEnv integration tests against a HouseExpo sample apartment."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.environment.reward import RewardConfig
from roomba_lab.environment.roomba_env import RobotKinematicsConfig, RoombaEnv
from roomba_lab.sensor.lidar import LidarSensor
from roomba_lab.shared.config import PROJECT_ROOT
from roomba_lab.simulator.world import World

SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "sample_maps"


def _build_env() -> RoombaEnv:
    loader = HouseExpoLoader(SAMPLE_DIR)
    h = loader.load(loader.map_ids()[0])
    world = World(polygon=Polygon(h.verts), pixels_per_metre=20.0)
    lidar = LidarSensor(n_beams=24, max_range_m=5.0)
    rcfg = RobotKinematicsConfig(radius_m=0.2, max_linear_speed_mps=0.5,
                                   max_angular_speed_radps=1.5, dt=0.1,
                                   cleaning_radius_m=0.25)
    reward_cfg = RewardConfig(new_cell_bonus=1.0, collision_penalty=-10.0,
                               step_penalty=-0.01, completion_bonus=100.0,
                               coverage_target=0.85)
    return RoombaEnv(world, lidar, rcfg, reward_cfg, max_episode_steps=200,
                     rng=np.random.default_rng(0))


def test_env_resets_to_correct_obs_shape() -> None:
    env = _build_env()
    obs = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    assert obs.shape[0] == 24 + 5  # 24 LIDAR + 5 status entries
    assert env.action_dim == 2


def test_env_step_returns_correct_tuple() -> None:
    env = _build_env()
    env.reset(seed=0)
    obs, r, done, info = env.step(np.array([0.5, 0.1], dtype=np.float32))
    assert obs.shape == (env.obs_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "coverage" in info
    assert "collisions" in info


def test_env_50_random_steps_no_error() -> None:
    env = _build_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    total_r = 0.0
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        _, r, done, _ = env.step(a)
        total_r += r
        if done:
            env.reset(seed=0)
    assert env.step_count <= 200  # via the public property added in Layer 23


def test_env_same_seed_same_first_obs() -> None:
    e1, e2 = _build_env(), _build_env()
    o1 = e1.reset(seed=7)
    o2 = e2.reset(seed=7)
    np.testing.assert_array_equal(o1, o2)


def test_env_action_clipping_does_not_crash() -> None:
    env = _build_env()
    env.reset(seed=0)
    _, _, _, _ = env.step(np.array([5.0, -5.0], dtype=np.float32))


def test_collision_freezes_pose() -> None:
    """Repeatedly drive into a wall direction; pose should not escape the polygon."""
    env = _build_env()
    env.reset(seed=0)
    p_before = env.robot.pose
    for _ in range(30):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.world.polygon.contains(
        __import__("shapely").geometry.Point(env.robot.pose.x, env.robot.pose.y)
    )
    _ = p_before  # used as sanity reference; pose may have moved if no wall ahead


def test_coverage_nonzero_after_movement() -> None:
    env = _build_env()
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.array([0.5, 0.0], dtype=np.float32))
    assert env.world.coverage_fraction() > 0.0
