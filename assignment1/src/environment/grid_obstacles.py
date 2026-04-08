"""Grid obstacle management utilities."""

import numpy as np
from typing import Tuple

try:
    from src.environment.grid_types import CellType
except ImportError:
    from environment.grid_types import CellType


class ObstacleManager:
    """Manages obstacles on the grid."""
    
    @staticmethod
    def add_obstacle(grid: np.ndarray, x: int, y: int,
                    grid_width: int, grid_height: int,
                    start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                    cell_type: CellType = CellType.BUILDING) -> bool:
        """Add obstacle to grid.
        
        Args:
            grid: Grid array.
            x: X coordinate.
            y: Y coordinate.
            grid_width: Grid width.
            grid_height: Grid height.
            start_pos: Start position (x, y).
            goal_pos: Goal position (x, y).
            cell_type: Type of cell to add.
            
        Returns:
            True if added successfully.
        """
        if 0 <= x < grid_width and 0 <= y < grid_height:
            if (x, y) == start_pos or (x, y) == goal_pos:
                return False
            grid[y, x] = cell_type.value
            return True
        return False
    
    @staticmethod
    def remove_obstacle(grid: np.ndarray, x: int, y: int,
                       grid_width: int, grid_height: int,
                       goal_pos: Tuple[int, int]) -> bool:
        """Remove obstacle from grid.
        
        Args:
            grid: Grid array.
            x: X coordinate.
            y: Y coordinate.
            grid_width: Grid width.
            grid_height: Grid height.
            goal_pos: Goal position (x, y).
            
        Returns:
            True if removed successfully.
        """
        if 0 <= x < grid_width and 0 <= y < grid_height:
            if (x, y) == goal_pos:
                return False
            grid[y, x] = CellType.EMPTY.value
            return True
        return False
