"""Main application for Grid-based Drone RL Simulator."""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import pygame
import numpy as np

from src.environment.grid_env import GridDroneEnv
from src.environment.grid_types import CellType
from src.rl.qlearning_agent import QLearningAgent
from src.visualization.grid_renderer import GridRenderer
from src.utils.config import Config
from src.app.event_handler import EventHandler
from src.app.training_loop import TrainingLoop
from src.app.save_load import SaveLoadManager


class GridApplication:
    """Main application for grid-based drone RL simulator."""
    
    def __init__(self, config_dir: Optional[Path] = None, load_model: Optional[Path] = None, 
                 grid_size: tuple = (15, 15)):
        """Initialize application."""
        print("Initializing Grid-based Drone RL Simulator...")
        
        self.config = Config(config_dir if config_dir else Path("configs"))
        
        # Environment
        env_config = {'grid': {'width': grid_size[0], 'height': grid_size[1]}, 'episode': {'max_steps': 200},
                     'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0, 'collision_penalty': -50.0,
                                'trap_penalty': -30.0, 'time_penalty': -0.1, 'wind_penalty': -0.5}}
        self.env = GridDroneEnv(env_config)
        
        # Agent
        agent_config = {'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
                       'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}}
        self.agent = QLearningAgent(self.env.observation_space.shape[0],
                                     self.env.action_space.n, agent_config)
        
        # Renderer and helpers
        self.renderer = GridRenderer(1400, 900, {})
        self.event_handler = EventHandler(self)
        self.training_loop = TrainingLoop(self)
        self.save_load_manager = SaveLoadManager(self)
        
        # State
        self.current_state, _ = self.env.reset()
        self.running, self.training_active, self.fast_forward = True, False, False
        self.show_heatmap, self.selected_tool = True, None
        
        # Stats
        self.episode, self.total_episodes = 0, 10000
        self.episode_reward, self.episode_steps = 0.0, 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count = 0
        
        # Timing
        self.clock, self.target_fps = pygame.time.Clock(), 30
        
        if load_model:
            self.agent.load(load_model)
        
        print(f"\n{'='*60}\nGrid Simulator Ready!\n  Grid: {grid_size[0]}x{grid_size[1]}")
        print(f"  Episode limit: {self.total_episodes}\nPress SPACE to start training!\n{'='*60}\n")
    
    def run(self) -> None:
        """Main application loop."""
        frame_count = 0
        while self.running:
            self.event_handler.handle_events()
            if self.training_active:
                self.training_loop.training_step()
            if not self.fast_forward or frame_count % 10 == 0:
                self.renderer.render(self.env, self.episode, self.episode_reward,
                    self.agent.epsilon, self.episode_steps, self.training_active,
                    self.selected_tool is not None, self.selected_tool, self.show_heatmap)
            self.clock.tick(1000 if self.fast_forward else self.target_fps)
            frame_count += 1
        self.renderer.cleanup()
    
    def reset_game(self) -> None:
        """Reset entire game without saving."""
        print(f"\n{'='*60}\n[RESET] Resetting EVERYTHING (no save)...")
        print(f"  Old Q-table: {len(self.agent.q_table)} states, episodes: {self.episode}")
        
        agent_config = {'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
                       'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}}
        self.agent = QLearningAgent(self.env.observation_space.shape[0],
                                     self.env.action_space.n, agent_config)
        
        self.episode, self.success_count = 0, 0
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.renderer.goal_rate = 0.0
        self.renderer.dashboard_panel.reward_history.clear()
        self.env.reset_heatmap()
        self.show_heatmap, self.training_active = False, False
        self.current_state, _ = self.env.reset()
        self.episode_reward, self.episode_steps = 0.0, 0
        
        print(f"[RESET] Complete! Fresh start!\n{'='*60}\n")
        self.renderer.show_notification("RESET COMPLETE!")
    
    def save_agent(self):
        self.save_load_manager.save_agent()
    
    def load_agent(self):
        self.save_load_manager.load_agent()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Grid-based Drone RL Simulator')
    parser.add_argument('--config', type=str, help='Config directory path')
    parser.add_argument('--load', type=str, help='Load model from path')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[20, 20],
                       help='Grid dimensions (width height)')
    
    args = parser.parse_args()
    
    config_dir = Path(args.config) if args.config else None
    load_model = Path(args.load) if args.load else None
    
    try:
        app = GridApplication(config_dir, load_model, tuple(args.grid_size))
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
