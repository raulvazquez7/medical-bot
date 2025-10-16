"""
Centralized prompt loading with caching.
Loads prompts once from YAML and caches for performance.
"""

import yaml
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """
    Loads prompts from YAML configuration file with LRU cache.
    Only loads once and reuses the result.

    Returns:
        Dictionary containing all prompt configurations

    Raises:
        FileNotFoundError: If prompts.yaml is not found
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
