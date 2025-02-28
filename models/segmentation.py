"""Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head."""

from typing import cast

from torch import Tensor, nn

from config.config_schema import SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.head import build_head
from models.utils import dtype_matcher


class SegmentationModel(nn.Module):
    # pylint: disable=missing-function-docstring
    """Segmentation head that produces object masks."""

    def __init__(
        self,
        config: SegmentationModelConfig,
        backbone2d: BackboneModel,
        backbone3d: BackboneModel,
        fusion: FusionModel,
    ) -> None:
        super().__init__()
        self._backbone2d = backbone2d
        self._backbone3d = backbone3d
        self._fusion = fusion
        self._config = config
        self._segmentation_head = build_head(self._config.head_config)
        self._dtype_matchers = {
            "backbone2d_to_fusion": dtype_matcher(self._backbone2d.head, self._fusion.head),
            "backbone3d_to_fusion": dtype_matcher(self._backbone3d.head, self._fusion.head),
            "fusion_to_segmentation": dtype_matcher(self._fusion.head, self._segmentation_head),
        }

    @property
    def backbone2d(self) -> BackboneModel:
        return self._backbone2d

    @property
    def backbone3d(self) -> BackboneModel:
        return self._backbone3d

    @property
    def fusion(self) -> FusionModel:
        return self._fusion

    def forward(self, image: Tensor, pointcloud: Tensor) -> Tensor:
        feature_map_2d = self._backbone2d(image)
        feature_map_2d = self._dtype_matchers["backbone2d_to_fusion"](feature_map_2d)

        feature_map_3d = self._backbone3d(pointcloud)
        feature_map_3d = self._dtype_matchers["backbone3d_to_fusion"](feature_map_3d)

        fused = self._fusion([feature_map_2d, feature_map_3d])
        fused = self._dtype_matchers["fusion_to_segmentation"](fused)

        segmented = self._segmentation_head(fused)
        return cast(Tensor, segmented)
