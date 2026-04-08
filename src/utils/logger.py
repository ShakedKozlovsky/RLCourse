"""Metrics logging and tracking system."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class MetricsLogger:
    """Logger for training metrics and episode data."""
    
    def __init__(self, log_dir: Path):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_data: List[Dict[str, Any]] = []
        self.csv_path = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.json_path = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self._csv_initialized = False
        self._csv_fieldnames = [
            'episode', 'reward', 'steps', 'success', 'collisions', 
            'energy_used', 'epsilon', 'loss', 'distance_traveled'
        ]
    
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log data for a completed episode.
        
        Args:
            episode_data: Dictionary containing episode metrics.
        """
        self.episode_data.append(episode_data)
        self._write_to_csv(episode_data)
    
    def _write_to_csv(self, episode_data: Dict[str, Any]) -> None:
        """Write episode data to CSV file."""
        mode = 'a' if self._csv_initialized else 'w'
        
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, 
                                   extrasaction='ignore')
            
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            
            writer.writerow(episode_data)
    
    def save_json(self) -> None:
        """Save all episode data to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
    
    def get_last_n_episodes(self, n: int) -> List[Dict[str, Any]]:
        """Get data from last N episodes.
        
        Args:
            n: Number of recent episodes to return.
            
        Returns:
            List of episode data dictionaries.
        """
        return self.episode_data[-n:] if len(self.episode_data) >= n else self.episode_data
    
    def get_mean_reward(self, window_size: int = 100) -> float:
        """Calculate mean reward over recent episodes.
        
        Args:
            window_size: Number of recent episodes to average.
            
        Returns:
            Mean reward value.
        """
        recent = self.get_last_n_episodes(window_size)
        if not recent:
            return 0.0
        rewards = [ep['reward'] for ep in recent]
        return float(np.mean(rewards))
    
    def get_success_rate(self, window_size: int = 100) -> float:
        """Calculate success rate over recent episodes.
        
        Args:
            window_size: Number of recent episodes to consider.
            
        Returns:
            Success rate as fraction [0, 1].
        """
        recent = self.get_last_n_episodes(window_size)
        if not recent:
            return 0.0
        successes = sum(1 for ep in recent if ep.get('success', False))
        return successes / len(recent)
    
    def get_mean_episode_length(self, window_size: int = 100) -> float:
        """Calculate mean episode length over recent episodes.
        
        Args:
            window_size: Number of recent episodes to average.
            
        Returns:
            Mean number of steps per episode.
        """
        recent = self.get_last_n_episodes(window_size)
        if not recent:
            return 0.0
        lengths = [ep['steps'] for ep in recent]
        return float(np.mean(lengths))
    
    def get_statistics(self, window_size: int = 100) -> Dict[str, float]:
        """Get comprehensive statistics over recent episodes.
        
        Args:
            window_size: Number of recent episodes to analyze.
            
        Returns:
            Dictionary of statistics.
        """
        recent = self.get_last_n_episodes(window_size)
        if not recent:
            return {}
        
        rewards = [ep['reward'] for ep in recent]
        steps = [ep['steps'] for ep in recent]
        successes = [ep.get('success', False) for ep in recent]
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_steps': float(np.mean(steps)),
            'success_rate': sum(successes) / len(successes),
            'total_episodes': len(self.episode_data),
        }
    
    def clear(self) -> None:
        """Clear all logged data (keeps files)."""
        self.episode_data = []
    
    def __len__(self) -> int:
        """Return number of logged episodes."""
        return len(self.episode_data)
