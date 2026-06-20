"""Differential-drive unicycle kinematics — the *only* place forward simulation
of the robot pose happens. Pure function: same input → same output.

L09 § 3 establishes that physical actuators (motor torques, steering angles)
require continuous control; this kinematics step is the bottom of that stack."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float

    def replace(self, **kw: float) -> Pose:
        x = kw.get("x", self.x)
        y = kw.get("y", self.y)
        theta = kw.get("theta", self.theta)
        return Pose(x=x, y=y, theta=theta)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def step_unicycle(
    pose: Pose,
    action: tuple[float, float],
    dt: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> Pose:
    """Advance a unicycle by `dt` seconds under a normalised action in ``[-1, 1]^2``.

    Action component 0 → forward velocity (×max_linear_speed).
    Action component 1 → angular velocity (×max_angular_speed)."""
    v_norm, w_norm = float(action[0]), float(action[1])
    v_norm = max(-1.0, min(1.0, v_norm))
    w_norm = max(-1.0, min(1.0, w_norm))
    v = v_norm * max_linear_speed
    w = w_norm * max_angular_speed
    return Pose(
        x=pose.x + v * math.cos(pose.theta) * dt,
        y=pose.y + v * math.sin(pose.theta) * dt,
        theta=_wrap_angle(pose.theta + w * dt),
    )
