"""Layer 3 — closed-form LIDAR tests on a 10×10 square room."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from roomba_lab.sensor.lidar import LidarSensor
from roomba_lab.simulator.kinematics import Pose


@pytest.fixture
def square() -> Polygon:
    return Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])


def test_centre_of_square_cardinal_beams(square: Polygon) -> None:
    lidar = LidarSensor(n_beams=4, max_range_m=20.0)
    pose = Pose(5.0, 5.0, 0.0)
    beams = lidar.scan(pose, square)
    assert all(b > 0.0 for b in beams)
    for b in beams:
        assert b == pytest.approx(5.0 / 20.0, abs=1e-2)


def test_corner_diagonal_beam_longer(square: Polygon) -> None:
    lidar = LidarSensor(n_beams=8, max_range_m=20.0)
    pose = Pose(0.5, 0.5, math.pi / 4)
    beams = lidar.scan(pose, square)
    assert max(beams) > 0.6


def test_beams_clamped_to_unit_interval(square: Polygon) -> None:
    lidar = LidarSensor(n_beams=16, max_range_m=5.0)
    pose = Pose(5.0, 5.0, 0.0)
    beams = lidar.scan(pose, square)
    assert np.all((beams >= 0.0) & (beams <= 1.0))


def test_facing_wall_returns_short_forward_beam(square: Polygon) -> None:
    lidar = LidarSensor(n_beams=8, max_range_m=10.0)
    pose = Pose(5.0, 5.0, 0.0)
    beams = lidar.scan(pose, square)
    assert beams[0] == pytest.approx(0.5, abs=2e-2)


def test_lidar_n_beams_shape(square: Polygon) -> None:
    for n in (4, 12, 24):
        lidar = LidarSensor(n_beams=n, max_range_m=5.0)
        beams = lidar.scan(Pose(5.0, 5.0, 0.0), square)
        assert beams.shape == (n,)
