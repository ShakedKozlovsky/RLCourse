"""Spawn-pose rejection sampler — split out of `roomba_env.py` in Layer 30
to keep `RoombaEnv` under the V3 § 3.2 150-LOC cap."""

from __future__ import annotations

import numpy as np

from roomba_lab.simulator.collision import is_collision, point_in_polygon
from roomba_lab.simulator.kinematics import Pose
from roomba_lab.simulator.world import World

MAX_SPAWN_ATTEMPTS = 200


def sample_spawn_pose(world: World, robot_radius: float,
                       rng: np.random.Generator) -> Pose:
    """Uniform-random rejection sampling for a collision-free spawn pose.

    Tries up to `MAX_SPAWN_ATTEMPTS` times to find a pose strictly inside
    the polygon and with the robot disk fully contained. Raises if none
    found — would indicate a pathologically narrow apartment."""
    x_min, y_min = world.bbox_min
    x_max, y_max = world.bbox_max
    for _ in range(MAX_SPAWN_ATTEMPTS):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        theta = rng.uniform(-np.pi, np.pi)
        pose = Pose(x, y, float(theta))
        if point_in_polygon(pose, world.polygon) and not is_collision(
            pose, world.polygon, robot_radius
        ):
            return pose
    raise RuntimeError(
        f"Could not find a valid spawn pose after {MAX_SPAWN_ATTEMPTS} attempts"
    )
