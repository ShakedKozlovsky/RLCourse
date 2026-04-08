"""Base renderer configuration and colors."""

import pygame
from typing import Dict, Any


class RendererBase:
    """Base renderer with shared configuration and colors."""
    
    def __init__(self, width: int, height: int, config: Dict[str, Any]):
        """Initialize base renderer.
        
        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            config: Rendering configuration.
        """
        self.width = width
        self.height = height
        self.config = config
        
        pygame.init()
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('DroneRL - Smart City Drone Delivery')
        
        # Beautiful thematic colors
        self.colors = {
            # Background and grid
            'background': (18, 22, 32),
            'grid_line': (45, 52, 70),
            'empty': (28, 35, 50),
            
            # Buildings
            'building': (160, 82, 45),
            'building_accent': (139, 69, 19),
            
            # Traps
            'trap': (255, 87, 51),
            'trap_accent': (220, 20, 60),
            
            # Goal
            'goal': (50, 205, 50),
            'goal_glow': (144, 238, 144),
            
            # Wind zones
            'wind_zone': (100, 149, 237),
            'wind_zone_light': (135, 206, 250),
            
            # Drone
            'drone': (255, 215, 0),
            'drone_accent': (255, 165, 0),
            'propeller': (200, 200, 200),
            
            # Wind arrows
            'arrow': (176, 224, 230),
            
            # UI elements
            'text': (230, 240, 255),
            'dashboard_bg': (22, 27, 38),
            'reward_positive': (50, 255, 150),
            'reward_negative': (255, 99, 71),
            
            # Menu colors
            'menu_bg': (25, 30, 42),
            'menu_hover': (45, 55, 75),
            'menu_selected': (65, 105, 225),
            'menu_selected_glow': (100, 149, 237),
            'button_text': (240, 248, 255),
            'button_label': (150, 170, 200),
        }
        
        # Font setup
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
    
    def get_screen(self) -> pygame.Surface:
        """Get pygame screen surface.
        
        Returns:
            Pygame screen surface.
        """
        return self.screen
    
    def cleanup(self) -> None:
        """Cleanup renderer resources."""
        pygame.quit()
