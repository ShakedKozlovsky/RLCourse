"""Single place that constructs a RoombaEnv from the config."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from roomba_lab.data.houseexpo_loader import HouseExpoLoader
from roomba_lab.environment.reward import RewardConfig
from roomba_lab.environment.roomba_env import RobotKinematicsConfig, RoombaEnv
from roomba_lab.sensor.lidar import LidarSensor
from roomba_lab.shared.config import PROJECT_ROOT, ConfigManager
from roomba_lab.simulator.world import World

SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "sample_maps"


def build_env(cfg: ConfigManager, map_id: str | None = None,
              max_episode_steps: int | None = None,
              rng: np.random.Generator | None = None) -> RoombaEnv:
    loader = HouseExpoLoader(SAMPLE_DIR)
    mid = map_id or cfg.get("env.primary_map_id") or loader.map_ids()[0]
    if mid not in loader.map_ids():
        mid = loader.map_ids()[0]
    house = loader.load(mid)
    world = World(polygon=Polygon(house.verts),
                  pixels_per_metre=float(cfg.get("env.map_pixels_per_metre")))
    lidar = LidarSensor(n_beams=int(cfg.get("sensor.n_lidar_beams")),
                         max_range_m=float(cfg.get("sensor.lidar_max_range_m")),
                         fov_degrees=float(cfg.get("sensor.fov_degrees")))
    robot_cfg = RobotKinematicsConfig(
        radius_m=float(cfg.get("robot.radius_m")),
        max_linear_speed_mps=float(cfg.get("robot.max_linear_speed_mps")),
        max_angular_speed_radps=float(cfg.get("robot.max_angular_speed_radps")),
        dt=float(cfg.get("robot.dt")),
        cleaning_radius_m=float(cfg.get("robot.cleaning_radius_m")),
    )
    reward_cfg = RewardConfig(
        new_cell_bonus=float(cfg.get("reward.new_cell_bonus")),
        collision_penalty=float(cfg.get("reward.collision_penalty")),
        step_penalty=float(cfg.get("reward.step_penalty")),
        completion_bonus=float(cfg.get("reward.completion_bonus")),
        coverage_target=float(cfg.get("reward.coverage_target")),
        coverage_progress_coef=float(cfg.get("reward.coverage_progress_coef", 0.0)),
    )
    return RoombaEnv(world, lidar, robot_cfg, reward_cfg,
                     max_episode_steps=max_episode_steps
                                       or int(cfg.get("env.max_episode_steps")),
                     rng=rng or np.random.default_rng(int(cfg.get("seed"))))
