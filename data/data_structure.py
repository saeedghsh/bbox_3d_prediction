"""Data structures"""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class Frame:
    """Frame containing multimodal data"""

    rgb: np.ndarray | Tensor
    pc: np.ndarray | Tensor
    mask: np.ndarray | Tensor
    bbox3d: np.ndarray | Tensor

    def __post_init__(self):
        Frame._assert_member_type_consistency(self)

    @staticmethod
    def _assert_member_type_consistency(frame: "Frame"):
        member_types = [type(getattr(frame, f.name)) for f in fields(Frame)]
        if not len(set(member_types)) == 1:
            msg = ", ".join([f"{f.name}: {type(getattr(frame, f.name))}" for f in fields(Frame)])
            raise ValueError(f"inconsistent member types: {msg}")

    @staticmethod
    def as_tensor(frame: "Frame", device: Optional[torch.device] = None) -> "Frame":
        """Return frame with all attributes as Torch Tensors.

        Allows optional device specification.
        """
        Frame._assert_member_type_consistency(frame)

        # If already Tensors, move to the requested device
        if isinstance(frame.rgb, torch.Tensor):
            if device is not None:
                return Frame(
                    rgb=frame.rgb.to(device),
                    pc=frame.pc.to(device),
                    mask=frame.mask.to(device),
                    bbox3d=frame.bbox3d.to(device),
                )
            return frame  # Already tensors and no device change needed

        # Convert from NumPy to PyTorch, respecting the device
        return Frame(
            rgb=torch.tensor(frame.rgb, device=device).permute(2, 0, 1).float() / 255.0,
            pc=torch.tensor(frame.pc, device=device).float(),
            mask=torch.tensor(frame.mask, device=device).long(),
            bbox3d=torch.tensor(frame.bbox3d, device=device).float(),
        )

    @staticmethod
    def as_numpy(frame: "Frame") -> "Frame":
        """Return frame with all attributes as NumPy arrays."""
        Frame._assert_member_type_consistency(frame)
        if isinstance(frame.rgb, np.ndarray):
            return frame
        return Frame(
            rgb=frame.rgb.cpu().numpy(),
            pc=frame.pc.cpu().numpy(),
            mask=frame.mask.cpu().numpy(),
            bbox3d=frame.bbox3d.cpu().numpy(),
        )
