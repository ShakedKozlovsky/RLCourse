"""Tests for obstacle management module."""

import pytest
import numpy as np
from src.environment.grid_obstacles import ObstacleManager
from src.environment.grid_types import CellType


def test_add_obstacle_in_bounds():
    """Test adding obstacle within bounds."""
    grid = np.zeros((10, 10), dtype=np.int32)
    result = ObstacleManager.add_obstacle(
        grid, 5, 5, 10, 10, (0, 0), (9, 9), CellType.BUILDING
    )
    
    assert result == True
    assert grid[5, 5] == CellType.BUILDING.value


def test_add_obstacle_out_of_bounds():
    """Test adding obstacle outside bounds fails."""
    grid = np.zeros((10, 10), dtype=np.int32)
    result = ObstacleManager.add_obstacle(
        grid, 15, 15, 10, 10, (0, 0), (9, 9), CellType.BUILDING
    )
    
    assert result == False


def test_add_obstacle_at_goal_fails():
    """Test cannot add obstacle at goal."""
    grid = np.zeros((10, 10), dtype=np.int32)
    result = ObstacleManager.add_obstacle(
        grid, 9, 9, 10, 10, (0, 0), (9, 9), CellType.BUILDING
    )
    
    assert result == False


def test_remove_obstacle_valid():
    """Test removing obstacle."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 5] = CellType.BUILDING.value
    
    result = ObstacleManager.remove_obstacle(grid, 5, 5, 10, 10, (9, 9))
    
    assert result == True
    assert grid[5, 5] == CellType.EMPTY.value


def test_remove_goal_fails():
    """Test cannot remove goal."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 9] = CellType.GOAL.value
    
    result = ObstacleManager.remove_obstacle(grid, 9, 9, 10, 10, (9, 9))
    
    assert result == False
