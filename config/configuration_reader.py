"""This module reads the configuration file."""

from pathlib import Path
from typing import Any

import yaml


def read_config(config_path: Path) -> Any:
    """Return the configuration at the path."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
