"""Virtual 2-D LIDAR — casts N evenly-spaced rays from the robot pose against
the apartment polygon boundary and returns range readings normalised to [0, 1].

EX05 § 1 — "LIDAR-style virtual sensors against walls"."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon

from roomba_lab.simulator.kinematics import Pose


@dataclass
class LidarSensor:
    n_beams: int
    max_range_m: float
    fov_degrees: float = 360.0

    def scan(self, pose: Pose, polygon: Polygon) -> np.ndarray:
        """Return an `n_beams`-length array of normalised distances in [0, 1]."""
        readings = np.full(self.n_beams, 1.0, dtype=np.float64)
        boundary = polygon.boundary
        fov_rad = math.radians(self.fov_degrees)
        for k in range(self.n_beams):
            offset = -fov_rad / 2.0 + (k + 0.5) * fov_rad / self.n_beams if self.fov_degrees < 360.0 \
                else (2.0 * math.pi * k / self.n_beams)
            ang = pose.theta + offset
            tip_x = pose.x + self.max_range_m * math.cos(ang)
            tip_y = pose.y + self.max_range_m * math.sin(ang)
            ray = LineString([(pose.x, pose.y), (tip_x, tip_y)])
            inter = ray.intersection(boundary)
            if inter.is_empty:
                readings[k] = 1.0
                continue
            readings[k] = _min_distance(pose, inter) / self.max_range_m
        return np.clip(readings, 0.0, 1.0)


def _min_distance(pose: Pose, intersection) -> float:
    """Return the smallest distance from `pose` to any point in `intersection`."""
    coords = _flatten_coords(intersection)
    return min(math.hypot(x - pose.x, y - pose.y) for x, y in coords)


def _flatten_coords(geom) -> list[tuple[float, float]]:
    if hasattr(geom, "geoms"):
        out: list[tuple[float, float]] = []
        for g in geom.geoms:
            out.extend(_flatten_coords(g))
        return out
    if hasattr(geom, "coords"):
        return [(float(x), float(y)) for x, y in geom.coords]
    return [(float(geom.x), float(geom.y))]
