"""Grid panel renderer for drone navigation display."""

import pygame
import numpy as np

try:
    from src.environment.grid_env import GridDroneEnv
    from src.visualization.cell_renderer import CellRenderer
    from src.visualization.drone_renderer import DroneRenderer
except ImportError:
    from environment.grid_env import GridDroneEnv
    from visualization.cell_renderer import CellRenderer
    from visualization.drone_renderer import DroneRenderer


class GridPanel:
    """Renders the grid panel with cells, drone, and wind arrows."""
    
    def __init__(self, screen: pygame.Surface, colors: dict,
                 panel_width: int, panel_height: int):
        """Initialize grid panel.
        
        Args:
            screen: Pygame screen surface.
            colors: Color dictionary.
            panel_width: Panel width in pixels.
            panel_height: Panel height in pixels.
        """
        self.screen = screen
        self.colors = colors
        self.panel_width = panel_width
        self.panel_height = panel_height
        
        # Initialize renderers
        self.cell_renderer = CellRenderer(screen, colors)
        self.drone_renderer = DroneRenderer(screen, colors)
    
    def render(self, env: GridDroneEnv, show_heatmap: bool = False) -> None:
        """Render grid with all elements.
        
        Args:
            env: Grid environment.
            show_heatmap: Whether to show visit heatmap.
        """
        grid = env.get_grid()
        wind_grid = env.get_wind_grid()
        heatmap = env.get_visit_heatmap() if show_heatmap else None
        grid_height, grid_width = grid.shape
        
        heatmap_normalized = self._normalize_heatmap(
            heatmap, grid_height, grid_width
        ) if show_heatmap else None
        
        cell_size, offset_x, offset_y = self._calculate_layout(
            grid_width, grid_height
        )
        
        # Render grid cells
        for y in range(grid_height):
            for x in range(grid_width):
                self._render_cell(
                    grid[y, x], x, y, cell_size, offset_x, offset_y,
                    heatmap_normalized, show_heatmap
                )
                
                # Draw wind arrows
                wind = wind_grid.get((x, y))
                if wind and (wind.dx != 0 or wind.dy != 0):
                    self.cell_renderer.draw_arrow(
                        offset_x + x * cell_size + cell_size // 2,
                        offset_y + y * cell_size + cell_size // 2,
                        wind.dx, wind.dy, cell_size // 3
                    )
        
        # Render drone
        self.drone_renderer.render_drone(
            env.drone_x, env.drone_y, cell_size, offset_x, offset_y
        )
    
    def _normalize_heatmap(self, heatmap, height: int, width: int):
        """Normalize heatmap values."""
        if heatmap is None:
            return None
        
        try:
            max_visits = int(np.max(heatmap))
            if max_visits > 0:
                normalized = np.zeros_like(heatmap, dtype=float)
                for y in range(height):
                    for x in range(width):
                        normalized[y, x] = float(heatmap[y, x]) / float(max_visits)
                return normalized
        except Exception as e:
            print(f"[HEATMAP] Normalization error: {e}")
        
        return None
    
    def _calculate_layout(self, grid_width: int, grid_height: int):
        """Calculate cell size and offsets."""
        padding = 40
        available_width = self.panel_width - 2 * padding
        available_height = self.panel_height - 2 * padding
        
        cell_size = min(
            available_width // grid_width,
            available_height // grid_height
        )
        
        grid_pixel_width = cell_size * grid_width
        grid_pixel_height = cell_size * grid_height
        
        offset_x = padding + (available_width - grid_pixel_width) // 2
        offset_y = padding + (available_height - grid_pixel_height) // 2
        
        return cell_size, offset_x, offset_y
    
    def _render_cell(self, cell_value, x: int, y: int,
                     cell_size: int, offset_x: int, offset_y: int,
                     heatmap_normalized, show_heatmap: bool) -> None:
        """Render a single grid cell."""
        rect = pygame.Rect(
            offset_x + x * cell_size,
            offset_y + y * cell_size,
            cell_size, cell_size
        )
        
        # Get heatmap color if applicable
        heatmap_color = None
        if show_heatmap and heatmap_normalized is not None:
            if (y < heatmap_normalized.shape[0] and
                x < heatmap_normalized.shape[1]):
                intensity = float(heatmap_normalized[y, x])
                if intensity > 0.01:
                    r = int(intensity * 255)
                    b = int((1 - intensity) * 255)
                    heatmap_color = (r, 0, b)
        
        # Render cell
        self.cell_renderer.render_cell(cell_value, rect, cell_size, heatmap_color)
