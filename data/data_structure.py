"""Data structures"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """Frame containing multimodal data"""

    rgb: np.ndarray
    pc: np.ndarray
    mask: np.ndarray
    bbox3d: np.ndarray
