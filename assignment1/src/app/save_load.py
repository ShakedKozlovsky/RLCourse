"""Save and load utilities for agent persistence."""

from pathlib import Path


class SaveLoadManager:
    """Manages save and load operations."""
    
    def __init__(self, app):
        """Initialize save/load manager.
        
        Args:
            app: Reference to main GridApplication.
        """
        self.app = app
        self.save_dir = Path("saved_models")
    
    def save_agent(self, filename: str = "agent.pkl") -> None:
        """Save agent to file."""
        print("\n" + "="*60)
        print("[SAVE] Saving agent...")
        
        self.save_dir.mkdir(exist_ok=True)
        save_path = self.save_dir / filename
        
        try:
            self.app.agent.save(save_path)
            print(f"[SUCCESS] Agent saved to: {save_path}")
            print(f"  Q-table size: {len(self.app.agent.q_table)} states")
            print(f"  Episodes: {self.app.episode}")
            print(f"  Epsilon: {self.app.agent.epsilon:.4f}")
            print("="*60 + "\n")
            self.app.renderer.show_notification(f"SAVED! Episode {self.app.episode}")
        except Exception as e:
            print(f"[ERROR] Failed to save: {e}")
            print("="*60 + "\n")
            self.app.renderer.show_notification("SAVE FAILED!")
    
    def load_agent(self, filename: str = "agent.pkl") -> None:
        """Load agent from file."""
        print("\n" + "="*60)
        print("[LOAD] Loading agent...")
        
        load_path = self.save_dir / filename
        
        if load_path.exists():
            try:
                self.app.agent.load(load_path)
                print(f"[SUCCESS] Agent loaded from: {load_path}")
                print(f"  Q-table size: {len(self.app.agent.q_table)} states")
                print(f"  Episodes: {self.app.agent.episodes}")
                print(f"  Epsilon: {self.app.agent.epsilon:.4f}")
                print("  You can now continue training or watch the agent!")
                print("="*60 + "\n")
                self.app.renderer.show_notification(
                    f"LOADED! {len(self.app.agent.q_table)} states"
                )
            except Exception as e:
                print(f"[ERROR] Failed to load: {e}")
                print("="*60 + "\n")
                self.app.renderer.show_notification("LOAD FAILED!")
        else:
            self._show_load_error(load_path)
    
    def _show_load_error(self, load_path: Path) -> None:
        """Show error when load file not found."""
        print(f"[NOT FOUND] No saved model at: {load_path}")
        print("  Available files in saved_models/:")
        
        if self.save_dir.exists():
            files = list(self.save_dir.glob("*.pkl"))
            if files:
                for f in files:
                    print(f"    - {f.name}")
            else:
                print("    (no saved models yet)")
        
        print("="*60 + "\n")
        self.app.renderer.show_notification("No saved model found!")
