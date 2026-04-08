"""Tests for reward calculation module."""

import pytest
from src.environment.grid_rewards import RewardCalculator
from src.environment.grid_types import CellType, Wind


def test_reward_calculator_initialization():
    """Test reward calculator initializes."""
    weights = {
        'progress': 1.0, 'goal': 100.0, 'collision': -50.0,
        'trap': -30.0, 'time': -0.1, 'wind': -0.5
    }
    calc = RewardCalculator(weights)
    
    assert calc.weights['goal'] == 100.0
    assert calc.weights['collision'] == -50.0


def test_goal_reward():
    """Test reaching goal gives correct reward."""
    weights = {
        'progress': 1.0, 'goal': 100.0, 'collision': -50.0,
        'trap': -30.0, 'time': -0.1, 'wind': -0.5
    }
    calc = RewardCalculator(weights)
    wind = Wind(0, 0)
    
    reward, terminated, collision_type = calc.calculate_step_reward(
        CellType.GOAL.value, 10.0, 0.0, wind, 5, 5, 5, 5
    )
    
    assert reward > 90
    assert terminated == True
    assert collision_type == 'goal'


def test_collision_penalty():
    """Test collision gives negative reward."""
    weights = {
        'progress': 1.0, 'goal': 100.0, 'collision': -50.0,
        'trap': -30.0, 'time': -0.1, 'wind': -0.5
    }
    calc = RewardCalculator(weights)
    wind = Wind(0, 0)
    
    reward, terminated, collision_type = calc.calculate_step_reward(
        CellType.BUILDING.value, 10.0, 9.0, wind, 5, 5, 3, 3
    )
    
    assert reward < -40
    assert terminated == True
    assert collision_type == 'building'


def test_trap_penalty():
    """Test trap gives penalty."""
    weights = {
        'progress': 1.0, 'goal': 100.0, 'collision': -50.0,
        'trap': -30.0, 'time': -0.1, 'wind': -0.5
    }
    calc = RewardCalculator(weights)
    wind = Wind(0, 0)
    
    reward, terminated, collision_type = calc.calculate_step_reward(
        CellType.TRAP.value, 10.0, 9.0, wind, 5, 5, 3, 3
    )
    
    assert reward < -20
    assert terminated == True
    assert collision_type == 'trap'


def test_progress_reward():
    """Test progress reward for moving closer."""
    weights = {
        'progress': 1.0, 'goal': 100.0, 'collision': -50.0,
        'trap': -30.0, 'time': -0.1, 'wind': -0.5
    }
    calc = RewardCalculator(weights)
    wind = Wind(0, 0)
    
    reward, terminated, collision_type = calc.calculate_step_reward(
        CellType.EMPTY.value, 10.0, 8.0, wind, 5, 5, 3, 3
    )
    
    assert reward > 0
    assert terminated == False
    assert collision_type is None
