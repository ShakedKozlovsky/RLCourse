"""2D Grid renderer for drone navigation - Main orchestrator."""

import pygame
from typing import Dict, Any

try:
    from src.environment.grid_env import GridDroneEnv
    from src.visualization.renderer_base import RendererBase
    from src.visualization.grid_panel import GridPanel
    from src.visualization.dashboard_panel import DashboardPanel
    from src.visualization.menu_panel import MenuPanel
    from src.visualization.notification_panel import NotificationPanel
except ImportError:
    from environment.grid_env import GridDroneEnv
    from visualization.renderer_base import RendererBase
    from visualization.grid_panel import GridPanel
    from visualization.dashboard_panel import DashboardPanel
    from visualization.menu_panel import MenuPanel
    from visualization.notification_panel import NotificationPanel


class GridRenderer(RendererBase):
    """2D grid-based renderer orchestrating all UI panels."""
    
    def __init__(self, width: int, height: int, config: Dict[str, Any]):
        """Initialize grid renderer.
        
        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            config: Rendering configuration.
        """
        super().__init__(width, height, config)
        
        # Layout configuration
        self.menu_height = 100
        self.grid_panel_width = int(width * 0.65)
        self.grid_panel_height = height - self.menu_height
        self.dashboard_x = self.grid_panel_width
        self.dashboard_width = width - self.grid_panel_width
        
        # Initialize panels
        self.grid_panel = GridPanel(
            self.screen, self.colors,
            self.grid_panel_width, self.grid_panel_height
        )
        
        self.dashboard_panel = DashboardPanel(
            self.screen, self.colors,
            self.dashboard_x, self.dashboard_width, height,
            self.font_large, self.font_medium, self.font_small
        )
        
        self.menu_panel = MenuPanel(
            self.screen, self.colors,
            width, height, self.menu_height,
            self.font_large, self.font_medium, self.font_small
        )
        
        self.notification_panel = NotificationPanel(
            self.screen, width, self.font_medium
        )
        
        # State tracking
        self.episode = 0
        self.total_reward = 0.0
        self.steps = 0
        self.goal_rate = 0.0
    
    def render(
        self,
        env: GridDroneEnv,
        episode: int,
        total_reward: float,
        epsilon: float,
        steps: int,
        training_active: bool = False,
        editor_mode: bool = False,
        selected_tool: str = None,
        show_heatmap: bool = False
    ) -> None:
        """Render complete frame.
        
        Args:
            env: Grid environment to render.
            episode: Current episode number.
            total_reward: Total reward for current episode.
            epsilon: Current epsilon value.
            steps: Current step number.
            training_active: Whether training is active.
            editor_mode: Whether editor mode is active.
            selected_tool: Currently selected tool name.
            show_heatmap: Whether to show heatmap.
        """
        self.episode = episode
        self.total_reward = total_reward
        self.steps = steps
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Render all panels
        self.grid_panel.render(env, show_heatmap)
        self.dashboard_panel.render(episode, total_reward, epsilon, steps,
                                   self.goal_rate)
        self.menu_panel.render(training_active, editor_mode, selected_tool)
        self.notification_panel.render()
        
        # Update display
        pygame.display.flip()
    
    def update_reward_history(self, reward: float) -> None:
        """Update reward history.
        
        Args:
            reward: Reward value to add.
        """
        self.dashboard_panel.update_reward_history(reward)
    
    def update_goal_rate(self, goal_rate: float) -> None:
        """Update goal success rate.
        
        Args:
            goal_rate: Goal success rate (0-1).
        """
        self.goal_rate = goal_rate
    
    def get_menu_click(self, pos: tuple) -> str:
        """Check if click was on menu button.
        
        Args:
            pos: Mouse position (x, y).
            
        Returns:
            Key string if button clicked, None otherwise.
        """
        return self.menu_panel.get_menu_click(pos)
    
    def show_notification(self, text: str) -> None:
        """Show notification message.
        
        Args:
            text: Notification text to display.
        """
        self.notification_panel.show(text)
