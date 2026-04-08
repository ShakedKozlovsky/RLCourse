"""Tests for grid environment module."""

import pytest
import numpy as np
from src.environment.grid_env import GridDroneEnv
from src.environment.grid_types import CellType


def test_environment_initialization():
    """Test environment initializes correctly."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    
    assert env.grid_width == 10
    assert env.grid_height == 10
    assert env.max_steps == 100
    assert env.action_space.n == 4
    assert env.observation_space.shape == (6,)


def test_environment_reset():
    """Test environment reset returns valid state."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    obs, info = env.reset()
    
    assert obs.shape == (6,)
    assert 'drone_position' in info
    assert 'goal_position' in info
    assert env.current_step == 0


def test_environment_step_valid_action():
    """Test step with valid action."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    env.reset()
    
    initial_pos = (env.drone_x, env.drone_y)
    obs, reward, terminated, truncated, info = env.step(0)
    
    assert obs.shape == (6,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'collision_type' in info


def test_add_obstacle_valid_position():
    """Test adding obstacle at valid position."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    
    result = env.add_obstacle(5, 5, CellType.BUILDING)
    
    assert result == True
    assert env.grid[5, 5] == CellType.BUILDING.value


def test_add_obstacle_at_start_fails():
    """Test adding obstacle at start position fails."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    
    result = env.add_obstacle(env.start_x, env.start_y, CellType.BUILDING)
    
    assert result == False


def test_remove_obstacle():
    """Test removing obstacle."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    
    env.add_obstacle(5, 5, CellType.BUILDING)
    result = env.remove_obstacle(5, 5)
    
    assert result == True
    assert env.grid[5, 5] == CellType.EMPTY.value


def test_heatmap_updates():
    """Test visit heatmap updates correctly."""
    config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    env = GridDroneEnv(config)
    env.reset()
    
    initial_heatmap = env.get_visit_heatmap().copy()
    env.step(1)
    updated_heatmap = env.get_visit_heatmap()
    
    assert not np.array_equal(initial_heatmap, updated_heatmap)
