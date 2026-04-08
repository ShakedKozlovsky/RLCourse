"""Reward calculation for grid environment."""

import numpy as np
from typing import Dict, Tuple

try:
    from src.environment.grid_types import CellType, Wind
except ImportError:
    from environment.grid_types import CellType, Wind


class RewardCalculator:
    """Calculates rewards for grid environment."""
    
    def __init__(self, reward_weights: Dict[str, float]):
        """Initialize reward calculator.
        
        Args:
            reward_weights: Dictionary of reward weights.
        """
        self.weights = reward_weights
    
    def calculate_step_reward(
        self,
        cell_value: int,
        prev_distance: float,
        current_distance: float,
        wind: Wind,
        goal_x: int,
        goal_y: int,
        new_x: int,
        new_y: int
    ) -> Tuple[float, bool, str]:
        """Calculate reward for a step.
        
        Args:
            cell_value: Value of destination cell.
            prev_distance: Previous distance to goal.
            current_distance: Current distance to goal.
            wind: Wind at current position.
            goal_x: Goal x position.
            goal_y: Goal y position.
            new_x: New x position.
            new_y: New y position.
            
        Returns:
            (reward, terminated, collision_type) tuple.
        """
        reward = self.weights['time']
        terminated = False
        collision_type = None
        
        if cell_value == CellType.BUILDING.value:
            reward += self.weights['collision']
            collision_type = 'building'
            terminated = True
        elif cell_value == CellType.TRAP.value:
            reward += self.weights['trap']
            collision_type = 'trap'
            terminated = True
        elif (cell_value == CellType.GOAL.value or 
              (new_x == goal_x and new_y == goal_y)):
            reward += self.weights['goal']
            terminated = True
            collision_type = 'goal'
        else:
            # Progress reward
            progress = prev_distance - current_distance
            reward += progress * self.weights['progress']
            
            # Wind penalty
            if hasattr(wind, 'strength') and wind.strength > 0.3:
                reward += self.weights['wind']
            elif abs(wind.dx) + abs(wind.dy) > 0:
                reward += self.weights['wind']
        
        return reward, terminated, collision_type
