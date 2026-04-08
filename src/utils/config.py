"""Simplified configuration management for grid simulator."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Simple configuration loader for grid simulator."""
    
    def __init__(self, config_dir: Path):
        """Initialize config from directory.
        
        Args:
            config_dir: Directory containing YAML config files.
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        
        if self.config_dir.exists():
            self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all YAML files from config directory."""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    config_name = config_file.stem
                    self.configs[config_name] = config_data
                    print(f"Loaded config: {config_name}")
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
    
    def get(self, section: str, default: Optional[Any] = None) -> Any:
        """Get configuration section.
        
        Args:
            section: Section name.
            default: Default value if not found.
            
        Returns:
            Configuration data or default.
        """
        return self.configs.get(section, default)
    
    def get_nested(self, *keys, default: Optional[Any] = None) -> Any:
        """Get nested configuration value.
        
        Args:
            *keys: Keys to navigate nested dict.
            default: Default value if not found.
            
        Returns:
            Configuration value or default.
        """
        value = self.configs
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if valid.
        """
        return True  # Simplified - no validation needed for grid simulator
