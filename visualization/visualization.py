"""This module provides a Visualizer class for visualizing images and point clouds"""

# pylint: disable=no-member
from typing import Any, Dict

import cv2
import numpy as np
import open3d as o3d

from data.data_structure import Frame


class Visualizer:  # pylint: disable=too-few-public-methods
    """Visualize images and point clouds from the dataset."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def _visualize_2d(self, frame: Frame) -> None:
        h, w, _ = frame.rgb.shape
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, (frame.mask.shape[0], 3), dtype=np.uint8)
        for i in range(frame.mask.shape[0]):
            mask_vis[frame.mask[i] > 0] = colors[i]

        stacked = np.hstack((frame.rgb, mask_vis))
        cv2.imshow("RGB & Mask", stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _visualize_3d(self, frame: Frame) -> None:
        points = frame.pc.reshape(3, -1).T
        colors = frame.rgb.reshape(-1, 3) / 255.0  # Normalize RGB for Open3D

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        lines = (
            [[i, (i + 1) % 4] for i in range(4)]
            + [[i + 4, (i + 1) % 4 + 4] for i in range(4)]
            + [[i, i + 4] for i in range(4)]
        )
        bboxes = []
        for box in frame.bbox3d:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            color = np.random.rand(3).tolist()
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
            bboxes.append(line_set)

        o3d.visualization.draw_geometries([pcd] + bboxes)

    def visualize_frame(self, frame: Frame) -> None:
        """Visualize a frame with its image, point cloud, and annotations."""
        if self._config["visualize_3d"]:
            self._visualize_3d(frame)
        if self._config["visualize_2d"]:
            self._visualize_2d(frame)
