"""Data structures"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Frame:
    """Frame containing multimodal data"""

    rgb: np.ndarray[Any, Any]
    pc: np.ndarray[Any, Any]
    mask: np.ndarray[Any, Any]
    bbox3d: np.ndarray[Any, Any]
