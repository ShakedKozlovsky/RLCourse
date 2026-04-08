"""Event handling for grid simulator application."""

import pygame
from pygame.locals import *

try:
    from src.environment.grid_types import CellType
except ImportError:
    from environment.grid_types import CellType


class EventHandler:
    """Handles pygame events and user input."""
    
    def __init__(self, app):
        """Initialize event handler.
        
        Args:
            app: Reference to main GridApplication.
        """
        self.app = app
    
    def handle_events(self) -> None:
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.app.running = False
            elif event.type == KEYDOWN:
                self.handle_keyboard(event.key)
            elif event.type == MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
    
    def handle_keyboard(self, key: int) -> None:
        """Handle keyboard input."""
        if key == K_ESCAPE:
            self.app.running = False
        elif key == K_SPACE:
            self.app.training_active = not self.app.training_active
            print(f">>> Training {'STARTED' if self.app.training_active else 'PAUSED'}")
        elif key == K_f:
            self.app.fast_forward = not self.app.fast_forward
            print(f">>> Fast forward: {'ON' if self.app.fast_forward else 'OFF'}")
        elif key == K_1:
            self.toggle_tool('building')
        elif key == K_2:
            self.toggle_tool('trap')
        elif key == K_3:
            self.toggle_tool('wind')
        elif key == K_x:
            self.toggle_tool('eraser')
        elif key == K_h:
            self.app.show_heatmap = not self.app.show_heatmap
        elif key == K_s:
            print("\n>>> SAVE button pressed!")
            self.app.save_agent()
        elif key == K_l:
            print("\n>>> LOAD button pressed!")
            self.app.load_agent()
        elif key == K_r:
            print("\n>>> RESET button pressed!")
            self.app.reset_game()
    
    def toggle_tool(self, tool: str) -> None:
        """Toggle tool selection."""
        if self.app.selected_tool == tool:
            self.app.selected_tool = None
            print(f">>> {tool.upper()} tool deselected")
        else:
            self.app.selected_tool = tool
            icons = {'building': '[BUILD]', 'trap': '[TRAP]',
                    'wind': '[WIND]', 'eraser': '[ERASE]'}
            print(f"> {icons.get(tool, '')} {tool.upper()} tool selected")
    
    def handle_mouse_click(self, pos: tuple) -> None:
        """Handle mouse clicks."""
        menu_key = self.app.renderer.get_menu_click(pos)
        if menu_key:
            self.handle_menu_button(menu_key)
            return
        
        if self.app.selected_tool:
            self.handle_grid_click(pos)
    
    def handle_menu_button(self, key: str) -> None:
        """Handle menu button clicks."""
        key_map = {
            'SPACE': K_SPACE, 'F': K_f, 'H': K_h, '1': K_1, '2': K_2,
            '3': K_3, 'X': K_x, 'R': K_r, 'S': K_s, 'L': K_l
        }
        if key in key_map:
            self.handle_keyboard(key_map[key])
    
    def handle_grid_click(self, pos: tuple) -> None:
        """Handle clicks on grid."""
        menu_y = self.app.renderer.height - self.app.renderer.menu_height
        if pos[1] >= menu_y:
            return
        
        if pos[0] >= self.app.renderer.grid_panel_width:
            return
        
        padding = 40
        available_width = self.app.renderer.grid_panel_width - 2 * padding
        available_height = self.app.renderer.grid_panel_height - 2 * padding
        
        cell_size = min(
            available_width // self.app.env.grid_width,
            available_height // self.app.env.grid_height
        )
        
        grid_pixel_width = cell_size * self.app.env.grid_width
        grid_pixel_height = cell_size * self.app.env.grid_height
        offset_x = padding + (available_width - grid_pixel_width) // 2
        offset_y = padding + (available_height - grid_pixel_height) // 2
        
        grid_x = (pos[0] - offset_x) // cell_size
        grid_y = (pos[1] - offset_y) // cell_size
        
        if 0 <= grid_x < self.app.env.grid_width and 0 <= grid_y < self.app.env.grid_height:
            self.apply_tool(grid_x, grid_y)
    
    def apply_tool(self, x: int, y: int) -> None:
        """Apply selected tool to grid."""
        if self.app.selected_tool == 'eraser':
            if self.app.env.remove_obstacle(x, y):
                self.app.renderer.show_notification(f"Removed obstacle at ({x}, {y})")
        elif self.app.selected_tool == 'building':
            if self.app.env.add_obstacle(x, y, CellType.BUILDING):
                self.app.renderer.show_notification(f"Building added at ({x}, {y})")
        elif self.app.selected_tool == 'trap':
            if self.app.env.add_obstacle(x, y, CellType.TRAP):
                self.app.renderer.show_notification(f"Trap added at ({x}, {y})")
        elif self.app.selected_tool == 'wind':
            if self.app.env.add_obstacle(x, y, CellType.WIND_ZONE):
                self.app.renderer.show_notification(f"Wind zone added at ({x}, {y})")
