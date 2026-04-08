"""Integration tests for complete system."""

import pytest
import numpy as np
from src.environment.grid_env import GridDroneEnv
from src.rl.qlearning_agent import QLearningAgent


def test_full_episode():
    """Test complete episode execution."""
    env_config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 50},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    agent_config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    
    env = GridDroneEnv(env_config)
    agent = QLearningAgent(6, 4, agent_config)
    
    state, _ = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 50:
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
        steps += 1
    
    assert steps > 0
    assert len(agent.q_table) > 0


def test_training_convergence():
    """Test agent learns over multiple episodes."""
    env_config = {
        'grid': {'width': 10, 'height': 10},
        'episode': {'max_steps': 100},
        'rewards': {'progress_weight': 1.0, 'goal_reward': 100.0,
                   'collision_penalty': -50.0, 'trap_penalty': -30.0,
                   'time_penalty': -0.1, 'wind_penalty': -0.5}
    }
    agent_config = {
        'hyperparameters': {'learning_rate': 0.1, 'gamma': 0.99, 'state_bins': 10},
        'exploration': {'initial_epsilon': 1.0, 'final_epsilon': 0.01, 'epsilon_decay': 0.995}
    }
    
    env = GridDroneEnv(env_config)
    agent = QLearningAgent(6, 4, agent_config)
    
    initial_q_size = len(agent.q_table)
    
    for episode in range(10):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            done = terminated or truncated
            steps += 1
        
        agent.decay_epsilon()
    
    assert len(agent.q_table) > initial_q_size
    assert agent.epsilon < 1.0
    assert agent.training_steps > 0
