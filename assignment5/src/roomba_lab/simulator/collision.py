"""Collision detection — checks whether a robot disk fits inside the apartment
polygon. ADR-002: shapely for the polygon ops, NumPy for the coverage raster."""

from __future__ import annotations

from shapely.geometry import Point, Polygon

from roomba_lab.simulator.kinematics import Pose


def is_collision(pose: Pose, polygon: Polygon, robot_radius: float) -> bool:
    """Return True if a disk of radius `robot_radius` at `pose` is NOT fully inside
    the apartment polygon (i.e. any part of the robot would clip a wall)."""
    disk = Point(pose.x, pose.y).buffer(robot_radius)
    return not polygon.contains(disk)


def point_in_polygon(pose: Pose, polygon: Polygon) -> bool:
    """Strict containment check — used by the random-spawn rejection sampler."""
    return polygon.contains(Point(pose.x, pose.y))
