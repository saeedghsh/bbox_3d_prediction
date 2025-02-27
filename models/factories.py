"""Factory module for instantiation of models."""

from typing import Optional

import torchvision.models as tv_models
from torchvision.models._api import WeightsEnum

from config.config_schema import BackboneModelConfig, FusionModelConfig, SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.segmentation import SegmentationModel


def _backbone_weights(config: BackboneModelConfig) -> Optional[WeightsEnum]:
    weights = None
    if config.pretrained:
        weights_enum = getattr(tv_models, f"{config.type}_Weights", None)
        weights = weights_enum.DEFAULT if weights_enum else None
    return weights


def _create_backbone_model(config: BackboneModelConfig) -> BackboneModel:
    """Return BackboneModel instance based on configuration."""
    if not (model_cls := getattr(tv_models, config.type, None)):
        raise ValueError(f"Invalid model name: {config.type}")
    backbone_instance = model_cls(weights=_backbone_weights(config))
    return BackboneModel(model=backbone_instance, config=config)


def create_segmentation_model(
    backbone_2d_config: BackboneModelConfig,
    backbone_3d_config: BackboneModelConfig,
    fusion_config: FusionModelConfig,
    segmentation_config: SegmentationModelConfig,
) -> SegmentationModel:
    """Return SegmentationModel instance based on configuration."""
    return SegmentationModel(
        segmentation_config=segmentation_config,
        backbone2d=_create_backbone_model(backbone_2d_config),
        backbone3d=_create_backbone_model(backbone_3d_config),
        fusion=FusionModel(fusion_config, backbone_2d_config, backbone_3d_config),
    )
