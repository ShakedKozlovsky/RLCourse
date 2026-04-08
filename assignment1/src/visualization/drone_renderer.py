"""Drone rendering utilities."""

import pygame


class DroneRenderer:
    """Renders the drone with propellers and effects."""
    
    def __init__(self, screen: pygame.Surface, colors: dict):
        """Initialize drone renderer.
        
        Args:
            screen: Pygame screen surface.
            colors: Color dictionary.
        """
        self.screen = screen
        self.colors = colors
    
    def render_drone(self, x: int, y: int, cell_size: int,
                    offset_x: int, offset_y: int) -> None:
        """Render drone at position.
        
        Args:
            x: Drone grid x position.
            y: Drone grid y position.
            cell_size: Size of grid cell.
            offset_x: Grid offset x.
            offset_y: Grid offset y.
        """
        cx = offset_x + x * cell_size + cell_size // 2
        cy = offset_y + y * cell_size + cell_size // 2
        radius = cell_size // 3
        
        # Shadow
        self._draw_shadow(cx, cy, radius)
        
        # Body
        self._draw_body(cx, cy, radius)
        
        # Arms
        self._draw_arms(cx, cy, radius)
        
        # Propellers
        self._draw_propellers(cx, cy, radius)
    
    def _draw_shadow(self, cx: int, cy: int, radius: int) -> None:
        """Draw drone shadow."""
        pygame.draw.circle(self.screen, (0, 0, 0, 50),
                         (cx + 2, cy + 2), radius + 1)
    
    def _draw_body(self, cx: int, cy: int, radius: int) -> None:
        """Draw drone body."""
        pygame.draw.circle(self.screen, self.colors['drone'],
                         (cx, cy), radius)
        pygame.draw.circle(self.screen, self.colors['drone_accent'],
                         (cx, cy), radius - 3)
        pygame.draw.circle(self.screen, (255, 255, 200),
                         (cx - 2, cy - 2), radius // 3)
    
    def _draw_arms(self, cx: int, cy: int, radius: int) -> None:
        """Draw propeller arms."""
        arm_len = int(radius * 1.5)
        arm_col = self.colors['propeller']
        
        pygame.draw.line(self.screen, arm_col,
                        (cx - arm_len, cy), (cx + arm_len, cy), 3)
        pygame.draw.line(self.screen, arm_col,
                        (cx, cy - arm_len), (cx, cy + arm_len), 3)
    
    def _draw_propellers(self, cx: int, cy: int, radius: int) -> None:
        """Draw propellers."""
        arm_len = int(radius * 1.5)
        
        for dx, dy in [(-arm_len, 0), (arm_len, 0),
                       (0, -arm_len), (0, arm_len)]:
            pygame.draw.circle(self.screen, (60, 60, 80),
                             (cx + dx, cy + dy), 5)
            pygame.draw.circle(self.screen, (180, 180, 200),
                             (cx + dx, cy + dy), 5, 2)
