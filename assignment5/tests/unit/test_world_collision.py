"""Layer 2 — World rasterisation + collision tests on a synthetic square world."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from roomba_lab.simulator.collision import is_collision, point_in_polygon
from roomba_lab.simulator.kinematics import Pose
from roomba_lab.simulator.world import OBSTACLE, UNVISITED, VISITED, World


@pytest.fixture
def square_world() -> World:
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    return World(polygon=poly, pixels_per_metre=10.0)


def test_world_grid_has_correct_dimensions(square_world: World) -> None:
    assert square_world.grid.shape == (40, 40)


def test_world_interior_is_unvisited(square_world: World) -> None:
    interior = square_world.grid[5:35, 5:35]
    assert np.all(interior == UNVISITED)


def test_world_borders_are_inside(square_world: World) -> None:
    free_count = square_world.free_cell_count()
    assert free_count > 1500


def test_cell_index_roundtrip(square_world: World) -> None:
    i, j = square_world.cell_index(2.0, 2.0)
    assert 18 <= i <= 21
    assert 18 <= j <= 21


def test_coverage_fraction_starts_at_zero(square_world: World) -> None:
    assert square_world.coverage_fraction() == 0.0


def test_coverage_fraction_after_marking_cells(square_world: World) -> None:
    free = square_world.free_cell_count()
    square_world.grid[(square_world.grid == UNVISITED)][:100]
    square_world.grid[10, 10] = VISITED
    square_world.grid[10, 11] = VISITED
    expected = 2.0 / free
    assert square_world.coverage_fraction() == pytest.approx(expected)


def test_collision_inside_polygon_safe(square_world: World) -> None:
    pose = Pose(2.0, 2.0, 0.0)
    assert not is_collision(pose, square_world.polygon, robot_radius=0.2)


def test_collision_outside_polygon_collides(square_world: World) -> None:
    pose = Pose(-1.0, 2.0, 0.0)
    assert is_collision(pose, square_world.polygon, robot_radius=0.2)


def test_collision_near_wall_collides(square_world: World) -> None:
    pose = Pose(0.05, 2.0, 0.0)
    assert is_collision(pose, square_world.polygon, robot_radius=0.2)


def test_point_in_polygon_true_for_centre(square_world: World) -> None:
    assert point_in_polygon(Pose(2.0, 2.0, 0.0), square_world.polygon)


def test_point_in_polygon_false_for_outside(square_world: World) -> None:
    assert not point_in_polygon(Pose(5.0, 5.0, 0.0), square_world.polygon)


def test_reset_visits_clears_visited_only(square_world: World) -> None:
    square_world.grid[5, 5] = VISITED
    square_world.grid[0, 0] = OBSTACLE
    square_world.reset_visits()
    assert square_world.grid[5, 5] == UNVISITED
    assert square_world.grid[0, 0] == OBSTACLE
