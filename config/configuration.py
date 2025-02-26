"""This module reads the configuration file."""

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def is_colab() -> bool:
    """Return True if the code is running in Google Colab."""
    return "google.colab" in sys.modules


def _resolve_colab_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the paths in the configuration file for Google Colab."""
    for key, value in config.items():
        if isinstance(value, dict) and set(value.keys()) == {"local", "colab"}:
            config[key] = value["colab"] if is_colab() else value["local"]
    return config


def read_config(config_path: Path) -> Any:
    """Return the configuration at the path."""
    with open(config_path.resolve(), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if isinstance(config, dict):
        config = _resolve_colab_paths(config)
    return config
