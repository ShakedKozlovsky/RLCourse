"""Dynamic robot state. Keeps a tiny history of poses for trajectory plots."""

from __future__ import annotations

from dataclasses import dataclass, field

from roomba_lab.simulator.kinematics import Pose


@dataclass
class Robot:
    pose: Pose
    radius: float
    cleaning_radius: float
    trajectory: list[Pose] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trajectory:
            self.trajectory.append(self.pose)

    def update(self, pose: Pose) -> None:
        self.pose = pose
        self.trajectory.append(pose)

    def reset(self, pose: Pose) -> None:
        self.pose = pose
        self.trajectory = [pose]
