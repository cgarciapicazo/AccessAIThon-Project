"""Utility functions and helper classes."""
import os
from pathlib import Path


def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root():
    """Get the project root directory.
    
    Returns:
        Path to project root.
    """
    return Path(__file__).parent.parent.parent


def load_config(config_path):
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    # TODO: Implement configuration loading
    pass
