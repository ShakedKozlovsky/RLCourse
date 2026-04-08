"""Q-table persistence utilities for saving and loading."""

import pickle
from pathlib import Path
from typing import Dict
import numpy as np


class QTablePersistence:
    """Handles Q-table saving and loading operations."""
    
    @staticmethod
    def save_qtable(
        q_table: Dict[tuple, np.ndarray],
        epsilon: float,
        episodes: int,
        training_steps: int,
        path: Path
    ) -> None:
        """Save Q-table and agent state to file.
        
        Args:
            q_table: Q-table dictionary.
            epsilon: Current epsilon value.
            episodes: Number of episodes trained.
            training_steps: Total training steps.
            path: Save path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'q_table': q_table,
            'epsilon': epsilon,
            'episodes': episodes,
            'training_steps': training_steps
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Q-table saved: {path}")
        print(f"  States: {len(q_table)}")
        print(f"  Episodes: {episodes}")
        print(f"  Epsilon: {epsilon:.4f}")
    
    @staticmethod
    def load_qtable(path: Path) -> Dict:
        """Load Q-table and agent state from file.
        
        Args:
            path: Load path.
            
        Returns:
            Dictionary with q_table, epsilon, episodes, training_steps.
        """
        if not path.exists():
            raise FileNotFoundError(f"Q-table file not found: {path}")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"Q-table loaded: {path}")
        print(f"  States: {len(save_data['q_table'])}")
        print(f"  Episodes: {save_data['episodes']}")
        print(f"  Epsilon: {save_data['epsilon']:.4f}")
        
        return save_data
