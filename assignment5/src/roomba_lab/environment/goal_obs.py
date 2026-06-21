"""Goal-conditioned observation augmenter — Layer 30, M1 substantive redesign.

The default `RoombaEnv._observation()` gives the agent LIDAR + (x, y, θ) +
coverage. That tells it about local walls and global progress, but NOT
where the nearest unvisited region is. With only that signal, the agent
cannot reliably navigate toward the frontier of cleaned vs unvisited area.

`augment_with_goal_direction(obs, env)` appends two normalised features:
   dx_to_nearest_unvisited, dy_to_nearest_unvisited
both in [-1, 1] relative to the apartment bounding-box span.

Adds 2 dims → obs_dim becomes 31 (vs 29 for the default env)."""

from __future__ import annotations

import numpy as np

from roomba_lab.environment.roomba_env import RoombaEnv
from roomba_lab.simulator.world import UNVISITED


def nearest_unvisited_direction(env: RoombaEnv) -> tuple[float, float]:
    """Return (dx, dy) normalised to bbox span pointing at the nearest
    unvisited cell from the robot's current pose. Returns (0, 0) if no
    unvisited cell remains (i.e. fully cleaned)."""
    grid = env.world.grid
    unvisited_mask = (grid == UNVISITED)
    if not unvisited_mask.any():
        return 0.0, 0.0
    i_robot, j_robot = env.world.cell_index(env.robot.pose.x, env.robot.pose.y)
    ii, jj = np.where(unvisited_mask)
    dists2 = (ii - i_robot) ** 2 + (jj - j_robot) ** 2
    k = int(np.argmin(dists2))
    i_t, j_t = int(ii[k]), int(jj[k])
    # Convert cell offset back to metres
    di_m = (i_t - i_robot) / env.world.pixels_per_metre
    dj_m = (j_t - j_robot) / env.world.pixels_per_metre
    x_span = max(1e-9, env.world.bbox_max[0] - env.world.bbox_min[0])
    y_span = max(1e-9, env.world.bbox_max[1] - env.world.bbox_min[1])
    return float(np.clip(dj_m / x_span, -1.0, 1.0)), float(np.clip(di_m / y_span, -1.0, 1.0))


class GoalConditionedEnv:
    """Thin wrapper around RoombaEnv that appends 2 frontier-direction features."""

    def __init__(self, env: RoombaEnv) -> None:
        self._env = env

    @property
    def obs_dim(self) -> int:
        """Default RoombaEnv obs_dim + 2 (dx, dy to nearest unvisited)."""
        return self._env.obs_dim + 2

    @property
    def action_dim(self) -> int:
        """Inherited from the wrapped env (action space is unchanged: 2D)."""
        return self._env.action_dim

    @property
    def world(self):
        """Forwarded access to the underlying World — needed by viz scripts."""
        return self._env.world

    @property
    def robot(self):
        """Forwarded access to the underlying Robot — needed for trajectory plots."""
        return self._env.robot

    @property
    def step_count(self) -> int:
        """Forwarded — the in-episode step counter of the wrapped env."""
        return self._env.step_count

    @property
    def collisions(self) -> int:
        """Forwarded — cumulative collision count of the wrapped env this episode."""
        return self._env.collisions

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the wrapped env and return the goal-augmented initial observation."""
        obs = self._env.reset(seed=seed)
        dx, dy = nearest_unvisited_direction(self._env)
        return np.concatenate([obs, [dx, dy]]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Step the wrapped env; append fresh (dx, dy) to the returned observation."""
        obs, reward, done, info = self._env.step(action)
        dx, dy = nearest_unvisited_direction(self._env)
        obs_aug = np.concatenate([obs, [dx, dy]]).astype(np.float32)
        return obs_aug, reward, done, info
