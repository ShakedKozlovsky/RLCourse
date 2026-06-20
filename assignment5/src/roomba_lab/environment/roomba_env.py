"""RoombaEnv — custom 2-D cleaning-robot environment.

**ADR-001**: This is NOT a `gym.Env` subclass. The spec § 1 forbids Gymnasium /
Gazebo dependence. We mirror the Gym API shape (`reset()`, `step(a) → (obs, r,
done, info)`) but import zero gym packages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomba_lab.environment.reward import RewardConfig, RewardInputs, compute_reward
from roomba_lab.sensor.lidar import LidarSensor
from roomba_lab.simulator.collision import is_collision, point_in_polygon
from roomba_lab.simulator.kinematics import Pose, step_unicycle
from roomba_lab.simulator.robot import Robot
from roomba_lab.simulator.world import UNVISITED, VISITED, World


@dataclass(frozen=True)
class RobotKinematicsConfig:
    radius_m: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    dt: float
    cleaning_radius_m: float


class RoombaEnv:
    def __init__(
        self,
        world: World,
        lidar: LidarSensor,
        robot_cfg: RobotKinematicsConfig,
        reward_cfg: RewardConfig,
        max_episode_steps: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.world = world
        self.lidar = lidar
        self.robot_cfg = robot_cfg
        self.reward_cfg = reward_cfg
        self.max_episode_steps = int(max_episode_steps)
        self._rng = rng or np.random.default_rng(0)
        self._step_count = 0
        self._completion_fired = False
        self._collisions = 0
        self._episode_reward = 0.0
        self.robot = Robot(pose=Pose(0.0, 0.0, 0.0),
                           radius=robot_cfg.radius_m,
                           cleaning_radius=robot_cfg.cleaning_radius_m)

    @property
    def obs_dim(self) -> int:
        return self.lidar.n_beams + 5

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.world.reset_visits()
        self._step_count = 0
        self._completion_fired = False
        self._collisions = 0
        self._episode_reward = 0.0
        self.robot.reset(self._sample_spawn_pose())
        self._mark_cleaned(self.robot.pose)
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = (float(action[0]), float(action[1]))
        coverage_before = self.world.coverage_fraction()
        candidate = step_unicycle(self.robot.pose, a, self.robot_cfg.dt,
                                   self.robot_cfg.max_linear_speed_mps,
                                   self.robot_cfg.max_angular_speed_radps)
        collided = is_collision(candidate, self.world.polygon, self.robot_cfg.radius_m)
        new_pose = self.robot.pose if collided else candidate
        self.robot.update(new_pose)
        new_cells = 0 if collided else self._mark_cleaned(new_pose)
        coverage_after = self.world.coverage_fraction()
        bonus_fired = (
            not self._completion_fired
            and coverage_before < self.reward_cfg.coverage_target <= coverage_after
        )
        reward, info = compute_reward(
            RewardInputs(new_cells, collided, coverage_before, coverage_after),
            self.reward_cfg,
        )
        if bonus_fired:
            self._completion_fired = True
        if collided:
            self._collisions += 1
        self._step_count += 1
        self._episode_reward += reward
        done = self._completion_fired or self._step_count >= self.max_episode_steps
        info.update({"coverage": coverage_after, "episode_reward": self._episode_reward,
                      "collisions": self._collisions, "step": self._step_count})
        return self._observation(), reward, done, info

    def _sample_spawn_pose(self) -> Pose:
        x_min, y_min = self.world.bbox_min
        x_max, y_max = self.world.bbox_max
        for _ in range(200):
            x = self._rng.uniform(x_min, x_max)
            y = self._rng.uniform(y_min, y_max)
            theta = self._rng.uniform(-np.pi, np.pi)
            pose = Pose(x, y, float(theta))
            if point_in_polygon(pose, self.world.polygon) and not is_collision(
                pose, self.world.polygon, self.robot_cfg.radius_m
            ):
                return pose
        raise RuntimeError("Could not find a valid spawn pose after 200 attempts")

    def _mark_cleaned(self, pose: Pose) -> int:
        radius_cells = max(1, int(self.robot_cfg.cleaning_radius_m * self.world.pixels_per_metre))
        i0, j0 = self.world.cell_index(pose.x, pose.y)
        h, w = self.world.grid.shape
        i_lo, i_hi = max(0, i0 - radius_cells), min(h, i0 + radius_cells + 1)
        j_lo, j_hi = max(0, j0 - radius_cells), min(w, j0 + radius_cells + 1)
        patch = self.world.grid[i_lo:i_hi, j_lo:j_hi]
        ii, jj = np.meshgrid(np.arange(i_lo, i_hi), np.arange(j_lo, j_hi), indexing="ij")
        dist2 = (ii - i0) ** 2 + (jj - j0) ** 2
        mask = (dist2 <= radius_cells ** 2) & (patch == UNVISITED)
        new = int(np.sum(mask))
        patch[mask] = VISITED
        return new

    def _observation(self) -> np.ndarray:
        beams = self.lidar.scan(self.robot.pose, self.world.polygon)
        x_min, y_min = self.world.bbox_min
        x_max, y_max = self.world.bbox_max
        x_norm = (self.robot.pose.x - x_min) / max(1e-9, x_max - x_min)
        y_norm = (self.robot.pose.y - y_min) / max(1e-9, y_max - y_min)
        sin_t = np.sin(self.robot.pose.theta)
        cos_t = np.cos(self.robot.pose.theta)
        cov = self.world.coverage_fraction()
        return np.concatenate([beams, [x_norm, y_norm, sin_t, cos_t, cov]]).astype(np.float32)
