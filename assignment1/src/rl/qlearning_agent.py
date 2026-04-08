"""Q-Learning Agent with Bellman updates and epsilon-greedy exploration."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from src.rl.qtable_persistence import QTablePersistence
except ImportError:
    from rl.qtable_persistence import QTablePersistence


class QLearningAgent:
    """Tabular Q-Learning agent using Bellman updates."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any]
    ):
        """Initialize Q-Learning agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            config: Configuration dictionary with hyperparameters.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        hyperparams = config.get('hyperparameters', {})
        self.learning_rate = hyperparams.get('learning_rate', 0.1)
        self.gamma = hyperparams.get('gamma', 0.99)
        
        exploration = config.get('exploration', {})
        self.initial_epsilon = exploration.get('initial_epsilon', 1.0)
        self.final_epsilon = exploration.get('final_epsilon', 0.01)
        self.epsilon_decay = exploration.get('epsilon_decay', 0.995)
        self.epsilon = self.initial_epsilon
        
        # Q-table: dictionary mapping state tuples to action values
        self.q_table: Dict[tuple, np.ndarray] = {}
        
        # State discretization parameters
        self.state_bins = hyperparams.get('state_bins', 10)
        
        self.training_steps = 0
        self.episodes = 0
        
        print(f"Q-Learning Agent initialized")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Epsilon: {self.epsilon} -> {self.final_epsilon}")
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for Q-table lookup."""
        discretized = np.clip(state, 0, self.state_bins - 1).astype(int)
        return tuple(discretized)
    
    def _get_q_values(self, state: tuple) -> np.ndarray:
        """Get Q-values for state, initialize if needed."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        return self.q_table[state]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        state_discrete = self._discretize_state(state)
        q_values = self._get_q_values(state_discrete)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        return int(np.argmax(q_values))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """Update Q-table using Bellman equation."""
        state_discrete = self._discretize_state(state)
        next_state_discrete = self._discretize_state(next_state)
        
        current_q = self._get_q_values(state_discrete)[action]
        
        if done:
            target_q = reward
        else:
            next_q_values = self._get_q_values(next_state_discrete)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.gamma * max_next_q
        
        td_error = target_q - current_q
        self.q_table[state_discrete][action] += self.learning_rate * td_error
        
        self.training_steps += 1
        
        return abs(td_error)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon value."""
        self.epsilon = max(self.final_epsilon, 
                          self.epsilon * self.epsilon_decay)
    
    def save(self, path: Path) -> None:
        """Save Q-table to file."""
        QTablePersistence.save_qtable(
            self.q_table, self.epsilon, self.episodes,
            self.training_steps, path
        )
    
    def load(self, path: Path) -> None:
        """Load Q-table from file."""
        save_data = QTablePersistence.load_qtable(path)
        self.q_table = save_data['q_table']
        self.epsilon = save_data['epsilon']
        self.episodes = save_data['episodes']
        self.training_steps = save_data['training_steps']
