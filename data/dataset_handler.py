"""Module for custom dataset class for multimodal data"""

# pylint: disable=no-member
import os
from typing import Callable, Dict, Optional
from dataclasses import dataclass, fields

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor


def _bbox3d_path(data_dir: str, frame_id: str) -> str:
    """Return the path to the 3D bounding box for the specified frame ID."""
    return os.path.join(data_dir, frame_id, "bbox3d.npy")


def _mask_path(data_dir: str, frame_id: str) -> str:
    """Return the path to the mask for the specified frame ID."""
    return os.path.join(data_dir, frame_id, "mask.npy")


def _pc_path(data_dir: str, frame_id: str) -> str:
    """Return the path to the point cloud for the specified frame ID."""
    return os.path.join(data_dir, frame_id, "pc.npy")


def _rgb_path(data_dir: str, frame_id: str) -> str:
    """Return the path to the RGB image for the specified frame ID."""
    return os.path.join(data_dir, frame_id, "rgb.jpg")


def read_image(file_path: str) -> np.ndarray:
    """Return the image for the given frame ID."""
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


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


class DatasetHandler(Dataset):
    """Custom dataset for multimodal data"""

    def __init__(self, data_dir, transform: Optional[Callable] = None):
        self._data_dir = data_dir
        self._transform = transform if transform else lambda x: x
        self._frame_ids = DatasetHandler._list_frame_ids(data_dir)
        self._verify_frames_files()

    @staticmethod
    def _list_frame_ids(path: str) -> list[str]:
        """Return a list of frame IDs in the specified directory.

        frame_ids are the names of the subdirectories in the data directory.
        """
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    def _verify_frames_files(self):
        """
        Verify that each frame contains the expected content.

        expected content: bbox3d.npy, mask.npy, pc.npy, rgb.jpg
        """
        for frame_id in self.frame_ids:
            os.path.isfile(_bbox3d_path(self._data_dir, frame_id))
            os.path.isfile(_mask_path(self._data_dir, frame_id))
            os.path.isfile(_pc_path(self._data_dir, frame_id))
            os.path.isfile(_rgb_path(self._data_dir, frame_id))

    @property
    def data_dir(self) -> str:
        """Return path to the dataset folder."""
        return self._data_dir

    @property
    def frame_ids(self) -> list[str]:
        """Return a list of frame IDs."""
        return self._frame_ids

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = self.frame_ids[idx]
        frame = Frame(
            rgb=read_image(_rgb_path(self.data_dir, frame_id)),
            pc=np.load(_pc_path(self.data_dir, frame_id)),
            mask=np.load(_mask_path(self.data_dir, frame_id)),
            bbox3d=np.load(_bbox3d_path(self.data_dir, frame_id)),
        )
        frame = self._transform(frame)
        return frame
