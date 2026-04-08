"""Grid environment setup utilities."""

import numpy as np
from typing import Dict, Tuple

try:
    from src.environment.grid_types import CellType, Wind
except ImportError:
    from environment.grid_types import CellType, Wind


class GridSetup:
    """Handles grid environment setup and initialization."""
    
    @staticmethod
    def setup_default_grid(grid: np.ndarray, grid_width: int, 
                          grid_height: int) -> Tuple[int, int, int, int]:
        """Setup default grid with obstacles.
        
        Args:
            grid: Grid array to modify.
            grid_width: Grid width.
            grid_height: Grid height.
            
        Returns:
            (start_x, start_y, goal_x, goal_y)
        """
        # Clear grid
        grid.fill(CellType.EMPTY.value)
        
        # Add buildings
        buildings = [(5, 7, 2, 1), (12, 2, 2, 2), (7, 12, 2, 2)]
        
        for x, y, w, h in buildings:
            for dx in range(w):
                for dy in range(h):
                    if 0 <= x + dx < grid_width and 0 <= y + dy < grid_height:
                        grid[y + dy, x + dx] = CellType.BUILDING.value
        
        # Add traps
        traps = [(8, 6), (15, 8)]
        for x, y in traps:
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid[y, x] = CellType.TRAP.value
        
        # Add wind zones
        wind_zones = [(3, 5, 2, 2), (10, 8, 2, 3)]
        
        for x, y, w, h in wind_zones:
            for dx in range(w):
                for dy in range(h):
                    if 0 <= x + dx < grid_width and 0 <= y + dy < grid_height:
                        grid[y + dy, x + dx] = CellType.WIND_ZONE.value
        
        # Set positions
        start_x, start_y = 0, 0
        goal_x = grid_width - 1
        goal_y = grid_height - 1
        grid[goal_y, goal_x] = CellType.GOAL.value
        
        return start_x, start_y, goal_x, goal_y
    
    @staticmethod
    def setup_wind(wind_grid: Dict[Tuple[int, int], Wind],
                   grid_width: int, grid_height: int) -> None:
        """Setup wind patterns.
        
        Args:
            wind_grid: Wind grid dictionary to populate.
            grid_width: Grid width.
            grid_height: Grid height.
        """
        for y in range(grid_height):
            for x in range(grid_width):
                if (x + y) % 3 == 0:
                    wind_grid[(x, y)] = Wind(dx=1, dy=0)
                elif (x + y) % 3 == 1:
                    wind_grid[(x, y)] = Wind(dx=0, dy=1)
                elif (x + y) % 5 == 0:
                    wind_grid[(x, y)] = Wind(dx=1, dy=1)
                else:
                    wind_grid[(x, y)] = Wind(dx=0, dy=0)
