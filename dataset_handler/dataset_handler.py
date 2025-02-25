"""Module for custom dataset class for multimodal data"""

# pylint: disable=no-member
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class FilePaths:
    """Class to manage file paths for multimodal data"""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def bbox3d(self, frame_id: str) -> Path:
        """Return the path to the 3D bounding box for the specified frame ID."""
        return self._data_dir / frame_id / "bbox3d.npy"

    def mask(self, frame_id: str) -> Path:
        """Return the path to the mask for the specified frame ID."""
        return self._data_dir / frame_id / "mask.npy"

    def pc(self, frame_id: str) -> Path:
        """Return the path to the point cloud for the specified frame ID."""
        return self._data_dir / frame_id / "pc.npy"

    def rgb(self, frame_id: str) -> Path:
        """Return the path to the RGB image for the specified frame ID."""
        return self._data_dir / frame_id / "rgb.jpg"


@dataclass
class Frame:
    """Frame containing multimodal data"""

    rgb: np.ndarray
    pc: np.ndarray
    mask: np.ndarray
    bbox3d: np.ndarray


class DatasetHandler(Dataset):  # type: ignore[type-arg]
    """Custom dataset for multimodal data"""

    def __init__(
        self, config: Dict[str, Any], transform: Optional[Callable[[Frame], Frame]] = None
    ) -> None:
        self._config = config
        self._data_dir = Path(config["dataset_dir"]).resolve()
        self._file_paths = FilePaths(self._data_dir)

        self._transform = transform if transform else lambda x: x
        self._frame_ids = DatasetHandler._list_frame_ids(self._data_dir)
        self._verify_frames_files()

    @staticmethod
    def _list_frame_ids(path: Path) -> list[str]:
        """Return a list of frame IDs in the specified directory.

        frame_ids are the names of the subdirectories in the data directory.
        """
        return sorted([d.name for d in path.iterdir() if d.is_dir()])

    def _verify_frames_files(self) -> None:
        """
        Verify that each frame contains the expected content.

        expected content: bbox3d.npy, mask.npy, pc.npy, rgb.jpg
        """
        for frame_id in self._frame_ids:
            if not self._file_paths.bbox3d(frame_id).is_file():
                raise FileNotFoundError(f"Missing 3D bounding box for frame {frame_id}")
            if not self._file_paths.mask(frame_id).is_file():
                raise FileNotFoundError(f"Missing mask for frame {frame_id}")
            if not self._file_paths.pc(frame_id).is_file():
                raise FileNotFoundError(f"Missing point cloud for frame {frame_id}")
            if not self._file_paths.rgb(frame_id).is_file():
                raise FileNotFoundError(f"Missing RGB image for frame {frame_id}")

    def _frame(self, idx: int) -> Frame:
        """Return the frame at the specified index."""
        frame_id = self._frame_ids[idx]
        frame = Frame(
            rgb=cv2.imread(str(self._file_paths.rgb(frame_id))),
            pc=np.load(self._file_paths.pc(frame_id)),
            mask=np.load(self._file_paths.mask(frame_id)),
            bbox3d=np.load(self._file_paths.bbox3d(frame_id)),
        )
        frame = self._transform(frame)
        return frame

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the RGB image, point cloud, mask, and 3D bounding box for the specified index"""
        frame = self._frame(idx)
        return frame.rgb, frame.pc, frame.mask, frame.bbox3d
