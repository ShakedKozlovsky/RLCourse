"""Grid-based drone environment for reinforcement learning."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from src.environment.grid_types import CellType, Wind
    from src.environment.grid_setup import GridSetup
    from src.environment.grid_obstacles import ObstacleManager
    from src.environment.grid_rewards import RewardCalculator
except ImportError:
    from environment.grid_types import CellType, Wind
    from environment.grid_setup import GridSetup
    from environment.grid_obstacles import ObstacleManager
    from environment.grid_rewards import RewardCalculator


class GridDroneEnv(gym.Env):
    """2D grid-based drone navigation environment."""
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize grid environment."""
        super().__init__()
        
        # Configuration
        self.grid_width = config.get('grid', {}).get('width', 20)
        self.grid_height = config.get('grid', {}).get('height', 20)
        self.max_steps = config.get('episode', {}).get('max_steps', 200)
        
        reward_config = config.get('rewards', {})
        self.reward_weights = {
            'progress': reward_config.get('progress_weight', 1.0),
            'goal': reward_config.get('goal_reward', 100.0),
            'collision': reward_config.get('collision_penalty', -50.0),
            'trap': reward_config.get('trap_penalty', -30.0),
            'time': reward_config.get('time_penalty', -0.1),
            'wind': reward_config.get('wind_penalty', -0.5)
        }
        
        # Initialize
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        self.wind_grid: Dict[Tuple[int, int], Wind] = {}
        self.visit_heatmap = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        self.reward_calculator = RewardCalculator(self.reward_weights)
        
        # Spaces
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        self.observation_space = gym.spaces.Box(
            low=0, high=max(self.grid_width, self.grid_height),
            shape=(6,), dtype=np.float32
        )
        
        # Setup
        positions = GridSetup.setup_default_grid(
            self.grid, self.grid_width, self.grid_height
        )
        self.start_x, self.start_y, self.goal_x, self.goal_y = positions
        GridSetup.setup_wind(self.wind_grid, self.grid_width, self.grid_height)
        
        self.drone_x, self.drone_y = self.start_x, self.start_y
        self.current_step = 0
        self.prev_distance_to_goal = self._get_distance_to_goal()
    
    def add_obstacle(self, x: int, y: int, cell_type: CellType = CellType.BUILDING) -> bool:
        """Add obstacle to grid."""
        return ObstacleManager.add_obstacle(self.grid, x, y, self.grid_width, 
            self.grid_height, (self.start_x, self.start_y), (self.goal_x, self.goal_y), cell_type)
    
    def remove_obstacle(self, x: int, y: int) -> bool:
        """Remove obstacle from grid."""
        return ObstacleManager.remove_obstacle(self.grid, x, y, self.grid_width,
            self.grid_height, (self.goal_x, self.goal_y))
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.drone_x, self.drone_y = self.start_x, self.start_y
        self.current_step = 0
        self.prev_distance_to_goal = self._get_distance_to_goal()
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        self.current_step += 1
        dx, dy = self.action_map[action]
        
        # Apply wind
        wind = self.wind_grid.get((self.drone_x, self.drone_y), Wind(0, 0))
        if np.random.random() < getattr(wind, 'strength', 0.3):
            dx += wind.dx
            dy += wind.dy
        
        # New position
        new_x = np.clip(self.drone_x + dx, 0, self.grid_width - 1)
        new_y = np.clip(self.drone_y + dy, 0, self.grid_height - 1)
        
        # Calculate reward
        current_distance = np.sqrt((new_x - self.goal_x)**2 + (new_y - self.goal_y)**2)
        reward, terminated, collision_type = self.reward_calculator.calculate_step_reward(
            self.grid[new_y, new_x], self.prev_distance_to_goal, current_distance,
            wind, self.goal_x, self.goal_y, new_x, new_y
        )
        
        # Update if valid
        if not terminated or collision_type == 'goal':
            self.drone_x, self.drone_y = new_x, new_y
            self.visit_heatmap[new_y, new_x] += 1
            self.prev_distance_to_goal = current_distance
        
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        info['collision_type'] = collision_type
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_distance_to_goal(self) -> float:
        """Calculate distance to goal."""
        return np.sqrt((self.drone_x - self.goal_x)**2 + (self.drone_y - self.goal_y)**2)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([self.drone_x, self.drone_y, self.goal_x, self.goal_y,
                        self.grid_width, self.grid_height], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {'drone_position': (self.drone_x, self.drone_y),
                'goal_position': (self.goal_x, self.goal_y),
                'distance_to_goal': self._get_distance_to_goal(),
                'current_step': self.current_step}
    
    def render(self) -> Optional[np.ndarray]:
        return None
    
    def get_grid(self) -> np.ndarray:
        return self.grid
    
    def get_wind_grid(self) -> Dict[Tuple[int, int], Wind]:
        return self.wind_grid
    
    def get_visit_heatmap(self) -> np.ndarray:
        return self.visit_heatmap
    
    def reset_heatmap(self) -> None:
        self.visit_heatmap.fill(0)
