"""Training loop logic for grid simulator."""

from typing import List


class TrainingLoop:
    """Manages training loop and episode management."""
    
    def __init__(self, app):
        """Initialize training loop.
        
        Args:
            app: Reference to main GridApplication.
        """
        self.app = app
    
    def training_step(self) -> None:
        """Perform one training step."""
        if self.app.current_state is None:
            return
        
        # Select and execute action
        action = self.app.agent.select_action(self.app.current_state, training=True)
        next_state, reward, terminated, truncated, info = self.app.env.step(action)
        
        # Update agent
        self.app.agent.update(
            self.app.current_state, action, reward,
            next_state, terminated or truncated
        )
        
        # Update stats
        self.app.episode_reward += reward
        self.app.episode_steps += 1
        self.app.current_state = next_state
        
        # Handle episode end
        if terminated or truncated:
            self.handle_episode_end(info, terminated)
    
    def handle_episode_end(self, info: dict, terminated: bool) -> None:
        """Handle end of episode."""
        self.app.episode_rewards.append(self.app.episode_reward)
        self.app.episode_lengths.append(self.app.episode_steps)
        self.app.renderer.update_reward_history(self.app.episode_reward)
        
        # Track success
        if info.get('collision_type') == 'goal':
            self.app.success_count += 1
            print(f"[SUCCESS] Goal reached! Episode {self.app.episode}: "
                  f"Reward={self.app.episode_reward:.1f}, Steps={self.app.episode_steps}")
        else:
            collision = info.get('collision_type', 'timeout')
            print(f"[FAIL] Episode {self.app.episode} ended ({collision}): "
                  f"Reward={self.app.episode_reward:.1f}, Steps={self.app.episode_steps}")
        
        # Update epsilon and goal rate
        self.app.agent.decay_epsilon()
        if self.app.episode_rewards:
            goal_rate = self.app.success_count / len(self.app.episode_rewards)
            self.app.renderer.update_goal_rate(goal_rate)
        
        # Periodic summary
        if self.app.episode > 0 and self.app.episode % 100 == 0:
            self.print_summary()
        
        # Next episode
        self.app.episode += 1
        self.reset_episode()
        
        # Check completion
        if self.app.episode >= self.app.total_episodes:
            print(f"\nTraining complete! {self.app.total_episodes} episodes finished.")
            self.app.training_active = False
            self.print_summary()
    
    def reset_episode(self) -> None:
        """Reset current episode."""
        self.app.current_state, _ = self.app.env.reset()
        self.app.episode_reward = 0.0
        self.app.episode_steps = 0
    
    def print_summary(self) -> None:
        """Print training summary."""
        if not self.app.episode_rewards:
            return
        
        avg_reward = sum(self.app.episode_rewards[-100:]) / min(100, len(self.app.episode_rewards))
        avg_length = sum(self.app.episode_lengths[-100:]) / min(100, len(self.app.episode_lengths))
        goal_rate = self.app.success_count / len(self.app.episode_rewards)
        
        print(f"\n{'='*60}")
        print(f"Episode {self.app.episode} Summary (last 100 episodes)")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Length: {avg_length:.1f} steps")
        print(f"  Goal Rate: {goal_rate:.1%} ({self.app.success_count}/{len(self.app.episode_rewards)})")
        print(f"  Q-table size: {len(self.app.agent.q_table)} states")
        print(f"  Epsilon: {self.app.agent.epsilon:.4f}")
        print(f"{'='*60}\n")
