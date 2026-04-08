"""Cell rendering utilities for grid display."""

import pygame
import numpy as np

try:
    from src.environment.grid_env import CellType
except ImportError:
    from environment.grid_env import CellType


class CellRenderer:
    """Renders individual cells with visual details."""
    
    def __init__(self, screen: pygame.Surface, colors: dict):
        """Initialize cell renderer.
        
        Args:
            screen: Pygame screen surface.
            colors: Color dictionary.
        """
        self.screen = screen
        self.colors = colors
    
    def render_cell(self, cell_value, rect, cell_size: int,
                   heatmap_color=None) -> None:
        """Render a single cell.
        
        Args:
            cell_value: Cell type value.
            rect: Cell rectangle.
            cell_size: Size of cell in pixels.
            heatmap_color: Optional heatmap color override.
        """
        # Determine color
        if heatmap_color:
            color = heatmap_color
        else:
            color = self._get_cell_color(cell_value)
        
        # Draw cell
        pygame.draw.rect(self.screen, color, rect)
        
        # Add details
        self._add_cell_details(cell_value, rect, cell_size)
        
        # Draw grid line
        pygame.draw.rect(self.screen, self.colors['grid_line'], rect, 1)
    
    def _get_cell_color(self, cell_value):
        """Get base color for cell type."""
        if cell_value == CellType.BUILDING.value:
            return self.colors['building']
        elif cell_value == CellType.TRAP.value:
            return self.colors['trap']
        elif cell_value == CellType.GOAL.value:
            return self.colors['goal']
        elif cell_value == CellType.WIND_ZONE.value:
            return self.colors['wind_zone']
        return self.colors['empty']
    
    def _add_cell_details(self, cell_value, rect, cell_size: int) -> None:
        """Add visual details to cells."""
        if cell_value == CellType.BUILDING.value:
            self._draw_building_windows(rect, cell_size)
        elif cell_value == CellType.TRAP.value:
            self._draw_trap_stripes(rect)
        elif cell_value == CellType.GOAL.value:
            pygame.draw.rect(self.screen, self.colors['goal_glow'],
                           rect.inflate(-4, -4), 3)
    
    def _draw_building_windows(self, rect, cell_size: int) -> None:
        """Draw building windows."""
        if cell_size > 15:
            window_size = max(2, cell_size // 8)
            for wx in range(2):
                for wy in range(2):
                    wx_pos = rect.x + (wx + 1) * rect.width // 3
                    wy_pos = rect.y + (wy + 1) * rect.height // 3
                    pygame.draw.rect(self.screen, (255, 248, 220),
                                   (wx_pos, wy_pos, window_size, window_size))
    
    def _draw_trap_stripes(self, rect) -> None:
        """Draw trap warning stripes."""
        for i in range(0, rect.width + rect.height, 8):
            pygame.draw.line(self.screen, self.colors['trap_accent'],
                           (rect.x + i, rect.y), (rect.x, rect.y + i), 2)
    
    def draw_arrow(self, x: int, y: int, dx: int, dy: int, length: int) -> None:
        """Draw wind direction arrow.
        
        Args:
            x: Arrow origin x.
            y: Arrow origin y.
            dx: Direction x.
            dy: Direction y.
            length: Arrow length.
        """
        if dx == 0 and dy == 0:
            return
        
        mag = np.sqrt(dx * dx + dy * dy)
        if mag > 0:
            dx, dy = dx / mag, dy / mag
        
        end_x, end_y = int(x + dx * length), int(y + dy * length)
        pygame.draw.line(self.screen, self.colors['arrow'],
                        (x, y), (end_x, end_y), 2)
        
        angle = np.arctan2(dy, dx)
        arrow_size = 5
        p1_x = end_x - arrow_size * np.cos(angle - np.pi / 6)
        p1_y = end_y - arrow_size * np.sin(angle - np.pi / 6)
        p2_x = end_x - arrow_size * np.cos(angle + np.pi / 6)
        p2_y = end_y - arrow_size * np.sin(angle + np.pi / 6)
        
        pygame.draw.polygon(self.screen, self.colors['arrow'],
                          [(end_x, end_y), (p1_x, p1_y), (p2_x, p2_y)])
