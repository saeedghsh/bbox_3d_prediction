"""Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head."""

from torch import Tensor, nn

from models.backbone import Backbone2DModel, Backbone3DModel
from models.fusion import FusionModel


class SegmentationModel(nn.Module):
    """Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head.

    Its forward pass processes input images and point clouds through the corresponding
    backbones, fuses their feature maps, and then applies a segmentation head.
    """

    def __init__(
        self,
        backbone2d: Backbone2DModel,
        backbone3d: Backbone3DModel,
        fusion: FusionModel,
        segmentation_out_channels: int,
    ) -> None:
        super().__init__()
        self._backbone2d = backbone2d
        self._backbone3d = backbone3d
        self._fusion = fusion
        self._segmentation_head = nn.Sequential(
            nn.Conv2d(
                self._fusion.out_channels, segmentation_out_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(segmentation_out_channels, segmentation_out_channels, kernel_size=1),
        )

    @property
    def backbone2d(self) -> Backbone2DModel:
        """Returns the 2D backbone model."""
        return self._backbone2d

    @property
    def backbone3d(self) -> Backbone3DModel:
        """Returns the 3D backbone model."""
        return self._backbone3d

    @property
    def fusion(self) -> FusionModel:
        """Returns the fusion model."""
        return self._fusion

    def forward(self, image: Tensor, pointcloud: Tensor) -> Tensor:
        """Forward pass."""
        feat2d: Tensor = self._backbone2d(image)
        feat3d: Tensor = self._backbone3d(pointcloud)
        fused: Tensor = self._fusion(feat2d, feat3d)
        seg_out: Tensor = self._segmentation_head(fused)
        return seg_out
