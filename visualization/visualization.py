"""This module provides a Visualizer class for visualizing images and point clouds"""

# pylint: disable=no-member
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import open3d as o3d

from data.data_structure import Frame


class Color:
    """Color utility class for drawing."""

    class Names(Enum):  # pylint: disable=missing-class-docstring
        RED = 0
        GREEN = 1
        BLUE = 2
        YELLOW = 3
        CYAN = 4
        MAGENTA = 5
        WHITE = 6
        BLACK = 7

    COLORS = {  # channel order: RGB
        Names.RED.value: (255, 0, 0),
        Names.GREEN.value: (0, 255, 0),
        Names.BLUE.value: (0, 0, 255),
        Names.YELLOW.value: (255, 255, 0),
        Names.CYAN.value: (0, 255, 255),
        Names.MAGENTA.value: (255, 0, 255),
        Names.WHITE.value: (255, 255, 255),
        Names.BLACK.value: (0, 0, 0),
    }

    @staticmethod
    def clip_to_unit(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Clip the color values to the range [0, 1]."""
        color_unit = tuple(int(c / 255.0) for c in color)
        return cast(Tuple[int, int, int], color_unit)  # Explicit cast to suppress mypy error

    @staticmethod
    def reorder_channels(color: Tuple[int, int, int], channel_order: str) -> Tuple[int, int, int]:
        """Reorder the color channels according to the given order."""
        channel_indices = {ch: i for i, ch in enumerate("rgb")}
        color_reordered = tuple(color[channel_indices[ch]] for ch in channel_order)
        return cast(Tuple[int, int, int], color_reordered)  # Explicit cast to suppress mypy error

    @staticmethod
    def color(color_in: Optional[int | Names] = None) -> Tuple[int, int, int]:
        """Return a color for drawing."""
        if color_in is None:
            return Color.COLORS[Color.Names.BLACK.value]
        if not isinstance(color_in, int) and not isinstance(color_in, Color.Names):
            raise ValueError(f"Invalid color type (should be [int|Color.Names]): {type(color_in)}")
        if isinstance(color_in, Color.Names):
            return Color.COLORS[color_in.value]
        return Color.COLORS[color_in % len(Color.COLORS)]

    @staticmethod
    def color_cv2(color_in: Optional[int | Names] = None) -> Tuple[int, int, int]:
        """Return a color for drawing in the OpenCV format."""
        color_out = Color.color(color_in)
        color_out = Color.reorder_channels(color_out, channel_order="bgr")
        return color_out

    @staticmethod
    def color_o3d(color_in: Optional[int | Names] = None) -> Tuple[int, int, int]:
        """Return a color for drawing in the Open3D format."""
        color_out = Color.color(color_in)
        color_out = Color.reorder_channels(color_out, channel_order="rgb")
        color_out = Color.clip_to_unit(color_out)
        return color_out

    @staticmethod
    def color_3d_axis() -> List[Tuple[int, int, int]]:
        """Return a list of colors for the 3D axes."""
        return [
            Color.color_o3d(Color.Names.RED),
            Color.color_o3d(Color.Names.GREEN),
            Color.color_o3d(Color.Names.BLUE),
        ]


def _draw_with_custom_camera_view(geometries: List[Any]) -> None:
    """Draw geometries with a custom camera view."""
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)
    # Set custom view aligned with camera Z-axis
    view_control = vis.get_view_control()
    camera = view_control.convert_to_pinhole_camera_parameters()
    # Set eye position slightly behind the origin, looking at [0, 0, 0], with Z-axis as up direction
    camera.extrinsic = np.array(
        [
            [1, 0, 0, 0],  # X-axis
            [0, 1, 0, 0],  # Y-axis
            [0, 0, 1, +1],  # Z-axis (set it behind the origin)
            [0, 0, 0, 1],
        ]
    )
    # Apply the new camera parameters
    view_control.convert_from_pinhole_camera_parameters(camera)
    # Run visualization
    vis.run()
    vis.destroy_window()


def _camera_frustum() -> o3d.geometry.LineSet:
    """Draw a pyramid for the camera"""
    pyramid = o3d.geometry.LineSet()
    near_plane = 0.10
    fov = 90.0  # vertical in degrees
    aspect_ratio = 16.0 / 9.0

    half_height_near = near_plane * np.tan(np.radians(fov / 2))
    half_width_near = half_height_near * aspect_ratio
    points = [
        [0, 0, 0],  # Camera origin
        [-half_width_near, -half_height_near, near_plane],  # Near plane vertex
        [half_width_near, -half_height_near, near_plane],  # Near plane vertex
        [half_width_near, half_height_near, near_plane],  # Near plane vertex
        [-half_width_near, half_height_near, near_plane],  # Near plane vertex
    ]
    lines = [
        [0, 1],  # Camera origin to near plane
        [0, 2],  # Camera origin to near plane
        [0, 3],  # Camera origin to near plane
        [0, 4],  # Camera origin to near plane
        [1, 2],  # Near plane edges
        [2, 3],  # Near plane edges
        [3, 4],  # Near plane edges
        [4, 1],  # Near plane edges
    ]
    colors = [Color.color_o3d(Color.Names.RED)] * len(lines)
    pyramid.points = o3d.utility.Vector3dVector(points)
    pyramid.lines = o3d.utility.Vector2iVector(lines)
    pyramid.colors = o3d.utility.Vector3dVector(colors)

    return pyramid


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

        geometries = [pcd] + bboxes + [_camera_frustum()]
        _draw_with_custom_camera_view(geometries)

    def visualize_frame(self, frame: Frame) -> None:
        """Visualize a frame with its image, point cloud, and annotations."""
        if self._config["visualize_3d"]:
            self._visualize_3d(frame)
        if self._config["visualize_2d"]:
            self._visualize_2d(frame)
