"""Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head."""

from typing import cast

from torch import Tensor, nn

from config.config_schema import SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel


class SegmentationModel(nn.Module):
    """Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head.

    Its forward pass processes input images and point clouds through the corresponding
    backbones, fuses their feature maps, and then applies a segmentation head.
    """

    def __init__(
        self,
        segmentation_config: SegmentationModelConfig,
        backbone2d: BackboneModel,
        backbone3d: BackboneModel,
        fusion: FusionModel,
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self._config = segmentation_config
        self._in_channels = fusion.config.out_channels
        self._out_channels = segmentation_config.out_channels

        self._backbone2d = backbone2d
        self._backbone3d = backbone3d
        self._fusion = fusion
        self._segmentation_head = nn.Sequential(
            nn.Conv2d(self._in_channels, self._out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self._out_channels, self._out_channels, kernel_size=1),
        )

    @property
    def backbone2d(self) -> BackboneModel:
        """Returns the 2D backbone model."""
        return self._backbone2d

    @property
    def backbone3d(self) -> BackboneModel:
        """Returns the 3D backbone model."""
        return self._backbone3d

    @property
    def fusion(self) -> FusionModel:
        """Returns the fusion model."""
        return self._fusion

    def forward(self, image: Tensor, pointcloud: Tensor) -> Tensor:
        """Forward pass."""
        feat2d = self._backbone2d(image)
        feat3d = self._backbone3d(pointcloud)
        fused = self._fusion(feat2d, feat3d)
        return cast(Tensor, self._segmentation_head(fused))
