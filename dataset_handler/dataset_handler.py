"""Module for custom dataset class for multimodal data"""

# pylint: disable=no-member
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


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
    return cv2.imread(file_path)


@dataclass
class Frame:
    """Frame containing multimodal data"""

    rgb: np.ndarray
    pc: np.ndarray
    mask: np.ndarray
    bbox3d: np.ndarray


class DatasetHandler(Dataset):  # type: ignore[type-arg]
    """Custom dataset for multimodal data"""

    def __init__(self, data_dir: str, transform: Optional[Callable[[Frame], Frame]] = None) -> None:
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

    def _verify_frames_files(self) -> None:
        """
        Verify that each frame contains the expected content.

        expected content: bbox3d.npy, mask.npy, pc.npy, rgb.jpg
        """
        for frame_id in self._frame_ids:
            if not os.path.isfile(_bbox3d_path(self._data_dir, frame_id)):
                raise FileNotFoundError(f"Missing 3D bounding box for frame {frame_id}")
            if not os.path.isfile(_mask_path(self._data_dir, frame_id)):
                raise FileNotFoundError(f"Missing mask for frame {frame_id}")
            if not os.path.isfile(_pc_path(self._data_dir, frame_id)):
                raise FileNotFoundError(f"Missing point cloud for frame {frame_id}")
            if not os.path.isfile(_rgb_path(self._data_dir, frame_id)):
                raise FileNotFoundError(f"Missing RGB image for frame {frame_id}")

    def _frame(self, idx: int) -> Frame:
        """Return the frame at the specified index."""
        frame_id = self._frame_ids[idx]
        frame = Frame(
            rgb=read_image(_rgb_path(self._data_dir, frame_id)),
            pc=np.load(_pc_path(self._data_dir, frame_id)),
            mask=np.load(_mask_path(self._data_dir, frame_id)),
            bbox3d=np.load(_bbox3d_path(self._data_dir, frame_id)),
        )
        frame = self._transform(frame)
        return frame

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the RGB image, point cloud, mask, and 3D bounding box for the specified index"""
        frame = self._frame(idx)
        return frame.rgb, frame.pc, frame.mask, frame.bbox3d
