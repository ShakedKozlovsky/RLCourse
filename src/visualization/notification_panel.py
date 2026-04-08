"""Notification panel for displaying temporary messages."""

import pygame
from typing import Optional


class NotificationPanel:
    """Renders notification popups with fade in/out effects."""
    
    def __init__(self, screen: pygame.Surface, width: int, font_medium):
        """Initialize notification panel.
        
        Args:
            screen: Pygame screen surface.
            width: Screen width.
            font_medium: Medium font.
        """
        self.screen = screen
        self.width = width
        self.font_medium = font_medium
        
        self.notification_text: Optional[str] = None
        self.notification_time: int = 0
        self.notification_duration: int = 3000  # 3 seconds in ms
    
    def show(self, text: str) -> None:
        """Show notification message.
        
        Args:
            text: Notification text.
        """
        self.notification_text = text
        self.notification_time = pygame.time.get_ticks()
    
    def render(self) -> None:
        """Render notification if active."""
        if self.notification_text is None:
            return
        
        # Check if notification expired
        current_time = pygame.time.get_ticks()
        if current_time - self.notification_time > self.notification_duration:
            self.notification_text = None
            return
        
        # Calculate fade
        alpha = self._calculate_alpha(current_time)
        
        # Draw notification box
        self._draw_notification_box(alpha)
    
    def _calculate_alpha(self, current_time: int) -> int:
        """Calculate alpha for fade effect.
        
        Args:
            current_time: Current time in ms.
            
        Returns:
            Alpha value (0-255).
        """
        elapsed = current_time - self.notification_time
        
        if elapsed < 300:
            # Fade in
            return int(255 * (elapsed / 300))
        elif elapsed > self.notification_duration - 300:
            # Fade out
            remaining = self.notification_duration - elapsed
            return int(255 * (remaining / 300))
        else:
            return 255
    
    def _draw_notification_box(self, alpha: int) -> None:
        """Draw notification box with alpha.
        
        Args:
            alpha: Alpha value (0-255).
        """
        box_width = 400
        box_height = 60
        box_x = (self.width - box_width) // 2
        box_y = 50
        
        # Create surface with alpha
        notification_surface = pygame.Surface((box_width, box_height))
        notification_surface.set_alpha(alpha)
        
        # Draw background
        pygame.draw.rect(notification_surface, (50, 50, 70),
                        (0, 0, box_width, box_height), border_radius=10)
        pygame.draw.rect(notification_surface, (100, 150, 255),
                        (0, 0, box_width, box_height), 3, border_radius=10)
        
        # Draw text
        text_surface = self.font_medium.render(self.notification_text, True,
                                               (255, 255, 255))
        text_rect = text_surface.get_rect(center=(box_width // 2, box_height // 2))
        notification_surface.blit(text_surface, text_rect)
        
        # Blit to screen
        self.screen.blit(notification_surface, (box_x, box_y))
