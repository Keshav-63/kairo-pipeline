import os
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / 'settings.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Settings file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()