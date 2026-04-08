"""Bottom menu panel for tool selection and actions."""

import pygame
from typing import List, Tuple, Optional


class MenuPanel:
    """Renders bottom menu with tools and action buttons."""
    
    def __init__(self, screen: pygame.Surface, colors: dict,
                 width: int, height: int, menu_height: int,
                 font_large, font_medium, font_small):
        """Initialize menu panel.
        
        Args:
            screen: Pygame screen surface.
            colors: Color dictionary.
            width: Screen width.
            height: Screen height.
            menu_height: Menu panel height.
            font_large: Large font.
            font_medium: Medium font.
            font_small: Small font.
        """
        self.screen = screen
        self.colors = colors
        self.width = width
        self.height = height
        self.menu_height = menu_height
        self.font_large = font_large
        self.font_medium = font_medium
        self.font_small = font_small
        
        self.menu_items: List[Tuple] = []
    
    def render(self, training_active: bool, editor_mode: bool,
               selected_tool: Optional[str]) -> None:
        """Render menu panel.
        
        Args:
            training_active: Whether training is active.
            editor_mode: Whether editor mode is active.
            selected_tool: Currently selected tool.
        """
        menu_y = self.height - self.menu_height
        
        # Background
        menu_rect = pygame.Rect(0, menu_y, self.width, self.menu_height)
        pygame.draw.rect(self.screen, self.colors['menu_bg'], menu_rect)
        pygame.draw.line(self.screen, self.colors['grid_line'],
                        (0, menu_y), (self.width, menu_y), 2)
        
        # Render buttons
        self._render_buttons(menu_y, training_active, selected_tool)
        
        # Render editor mode indicator
        if editor_mode:
            self._render_editor_indicator(menu_y, selected_tool)
    
    def _render_buttons(self, menu_y: int, training_active: bool,
                       selected_tool: Optional[str]) -> None:
        """Render menu buttons."""
        buttons = self._get_button_definitions(training_active)
        button_width, button_height = 120, 60
        button_spacing = 10
        start_x = 20
        y = menu_y + 20
        
        self.menu_items = []
        current_x = start_x
        
        for key, label, icon, tool_id in buttons:
            rect = pygame.Rect(current_x, y, button_width, button_height)
            self.menu_items.append((rect, key, tool_id))
            self._draw_button(rect, key, label, icon, tool_id, selected_tool)
            current_x += button_width + button_spacing
    
    def _get_button_definitions(self, training_active: bool):
        """Get button definitions."""
        return [
            ('SPACE', 'Training', 'PLAY' if not training_active else 'PAUSE', None),
            ('F', 'Fast Mode', 'FAST', None),
            ('H', 'Heatmap', 'HEAT', 'heatmap'),
            ('1', 'Building', 'BUILD', 'building'),
            ('2', 'Trap', 'TRAP', 'trap'),
            ('3', 'Wind', 'WIND', 'wind'),
            ('X', 'Eraser', 'ERASE', 'eraser'),
            ('R', 'Reset', 'RESET', None),
            ('S', 'Save', 'SAVE', None),
            ('L', 'Load', 'LOAD', None),
        ]
    
    def _draw_button(self, rect, key: str, label: str, icon: str,
                    tool_id: Optional[str], selected_tool: Optional[str]) -> None:
        """Draw a single button."""
        # Color
        if tool_id and selected_tool == tool_id:
            pygame.draw.rect(self.screen, self.colors['menu_selected_glow'],
                           rect.inflate(4, 4), border_radius=10)
            color = self.colors['menu_selected']
        elif rect.collidepoint(pygame.mouse.get_pos()):
            color = self.colors['menu_hover']
        else:
            color = (35, 42, 58)
        
        # Button box
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors['text'], rect, 2, border_radius=8)
        
        # Icon
        icon_surf = self.font_large.render(icon, True, self.colors['button_text'])
        self.screen.blit(icon_surf, icon_surf.get_rect(
            center=(rect.x + rect.width // 2, rect.y + 22)))
        
        # Label
        label_surf = self.font_small.render(label, True, self.colors['button_label'])
        self.screen.blit(label_surf, label_surf.get_rect(
            center=(rect.x + rect.width // 2, rect.y + 45)))
        
        # Key hint
        key_surf = self.font_small.render(f"[{key}]", True, (150, 150, 160))
        self.screen.blit(key_surf, key_surf.get_rect(
            center=(rect.x + rect.width // 2, rect.y + rect.height + 12)))
    
    def _render_editor_indicator(self, menu_y: int,
                                 selected_tool: Optional[str]) -> None:
        """Render editor mode indicator."""
        text = f">>> EDITOR MODE - Selected: {selected_tool or 'None'} <<<"
        surf = self.font_medium.render(text, True,
                                       self.colors['reward_positive'])
        rect = surf.get_rect(center=(self.width // 2, menu_y - 15))
        self.screen.blit(surf, rect)
    
    def get_menu_click(self, pos: tuple) -> Optional[str]:
        """Check if click was on menu button.
        
        Args:
            pos: Mouse position (x, y).
            
        Returns:
            Key string if button clicked, None otherwise.
        """
        for button_rect, key, tool_id in self.menu_items:
            if button_rect.collidepoint(pos):
                return key
        return None
