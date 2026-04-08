"""Dashboard panel for displaying metrics and reward history."""

import pygame
from typing import List


class DashboardPanel:
    """Renders the dashboard panel with metrics and charts."""
    
    def __init__(self, screen: pygame.Surface, colors: dict,
                 x: int, width: int, height: int,
                 font_large, font_medium, font_small):
        """Initialize dashboard panel.
        
        Args:
            screen: Pygame screen surface.
            colors: Color dictionary.
            x: Dashboard x position.
            width: Dashboard width.
            height: Dashboard height.
            font_large: Large font.
            font_medium: Medium font.
            font_small: Small font.
        """
        self.screen = screen
        self.colors = colors
        self.x = x
        self.width = width
        self.height = height
        self.font_large = font_large
        self.font_medium = font_medium
        self.font_small = font_small
        
        self.reward_history: List[float] = []
        self.max_history_length = 100
    
    def render(self, episode: int, total_reward: float,
               epsilon: float, steps: int, goal_rate: float) -> None:
        """Render dashboard.
        
        Args:
            episode: Current episode.
            total_reward: Total reward.
            epsilon: Epsilon value.
            steps: Current steps.
            goal_rate: Goal success rate.
        """
        # Background
        rect = pygame.Rect(self.x, 0, self.width, self.height)
        pygame.draw.rect(self.screen, self.colors['dashboard_bg'], rect)
        
        # Title
        title = self.font_large.render("DroneRL Dashboard", True,
                                       self.colors['text'])
        self.screen.blit(title, (self.x + 20, 30))
        
        # Metrics
        self._render_metrics(episode, total_reward, epsilon, steps, goal_rate)
        
        # Reward chart
        self._render_reward_chart()
        
        # Legend
        self._render_legend()
    
    def _render_metrics(self, episode: int, total_reward: float,
                       epsilon: float, steps: int, goal_rate: float) -> None:
        """Render metrics text."""
        y_offset = 100
        line_height = 30
        
        info_lines = [
            f"Episode: {episode}",
            f"Total Reward: {total_reward:.1f}",
            f"Epsilon: {epsilon:.4f}",
            f"Steps: {steps}",
            f"Goal Rate: {goal_rate:.1%}",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font_medium.render(line, True, self.colors['text'])
            self.screen.blit(text, (self.x + 20, y_offset + i * line_height))
    
    def _render_reward_chart(self) -> None:
        """Render reward history chart."""
        chart_y = 280
        chart_height = 150
        chart_width = self.width - 40
        
        title = self.font_medium.render("Reward History (last 100)", True,
                                        self.colors['text'])
        self.screen.blit(title, (self.x + 20, chart_y - 30))
        
        chart_rect = pygame.Rect(self.x + 20, chart_y, chart_width, chart_height)
        pygame.draw.rect(self.screen, (30, 30, 40), chart_rect)
        pygame.draw.rect(self.screen, self.colors['grid_line'], chart_rect, 1)
        
        if len(self.reward_history) > 1:
            self._draw_reward_line(chart_rect, chart_y, chart_width, chart_height)
    
    def _draw_reward_line(self, chart_rect, chart_y: int,
                         chart_width: int, chart_height: int) -> None:
        """Draw reward history line."""
        max_reward = max(self.reward_history)
        min_reward = min(self.reward_history)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1
        
        points = []
        for i, reward in enumerate(self.reward_history):
            x = self.x + 20 + (i / len(self.reward_history)) * chart_width
            normalized = (reward - min_reward) / reward_range
            y = chart_y + chart_height - (normalized * chart_height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.colors['reward_positive'],
                            False, points, 2)
    
    def _render_legend(self) -> None:
        """Render color legend."""
        legend_y = 470
        legend_items = [
            ("Empty", self.colors['empty']),
            ("Building", self.colors['building']),
            ("Trap", self.colors['trap']),
            ("Goal", self.colors['goal']),
            ("Wind Zone", self.colors['wind_zone']),
        ]
        
        for i, (label, color) in enumerate(legend_items):
            box_rect = pygame.Rect(self.x + 20, legend_y + i * 25, 15, 15)
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, self.colors['text'], box_rect, 1)
            
            label_surf = self.font_small.render(label, True, self.colors['text'])
            self.screen.blit(label_surf, (self.x + 45, legend_y + i * 25))
    
    def update_reward_history(self, reward: float) -> None:
        """Update reward history.
        
        Args:
            reward: Reward value to add.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history_length:
            self.reward_history.pop(0)
