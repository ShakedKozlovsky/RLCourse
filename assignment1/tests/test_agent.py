"""Tests for Q-Learning agent module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.rl.qlearning_agent import QLearningAgent


def test_agent_initialization():
    """Test agent initializes correctly."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    assert agent.learning_rate == 0.1
    assert agent.gamma == 0.99
    assert agent.epsilon == 1.0
    assert len(agent.q_table) == 0
    assert agent.episodes == 0


def test_state_discretization():
    """Test state discretization produces valid output."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    state = np.array([5.5, 10.2, 15.7, 8.3, 20.0, 20.0], dtype=np.float32)
    discrete = agent._discretize_state(state)
    
    assert isinstance(discrete, tuple)
    assert len(discrete) == 6
    assert all(0 <= d < 10 for d in discrete)


def test_action_selection_exploration():
    """Test action selection during exploration."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    state = np.array([1.0, 1.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    action = agent.select_action(state, training=True)
    
    assert 0 <= action < 4


def test_action_selection_exploitation():
    """Test action selection during exploitation."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 0.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    state = np.array([1.0, 1.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    action = agent.select_action(state, training=False)
    
    assert 0 <= action < 4


def test_q_table_update():
    """Test Q-table updates correctly."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    state = np.array([1.0, 1.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    next_state = np.array([2.0, 1.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    
    initial_q_size = len(agent.q_table)
    agent.update(state, 0, 1.0, next_state, False)
    
    assert len(agent.q_table) > initial_q_size
    assert agent.training_steps == 1


def test_epsilon_decay():
    """Test epsilon decays correctly."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    initial_epsilon = agent.epsilon
    agent.decay_epsilon()
    
    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= 0.01


def test_epsilon_minimum_bound():
    """Test epsilon respects minimum value."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 0.02, 'final_epsilon': 0.01, 'epsilon_decay': 0.5}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    for _ in range(10):
        agent.decay_epsilon()
    
    assert agent.epsilon == 0.01


def test_save_and_load():
    """Test agent save and load functionality."""
    config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
    
    state = np.array([1.0, 1.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    agent.update(state, 0, 1.0, state, False)
    agent.decay_epsilon()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_agent.pkl"
        agent.save(save_path)
        
        new_agent = QLearningAgent(state_dim=6, action_dim=4, config=config)
        new_agent.load(save_path)
        
        assert len(new_agent.q_table) == len(agent.q_table)
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.training_steps == agent.training_steps
