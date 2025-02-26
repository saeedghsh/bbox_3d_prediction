"""Factory module for instantiation of models."""

from dataclasses import asdict

import torchvision.models as tv_models

from config.config_schema import BackboneModelConfig, FusionModelConfig, SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.segmentation import SegmentationModel


def _create_backbone_model(config: BackboneModelConfig) -> BackboneModel:
    """Return BackboneModel instance based on configuration."""
    required_keys = {"type", "in_channels", "out_channels", "pretrained"}
    if not required_keys.issubset(asdict(config)):
        missing_keys = required_keys - asdict(config).keys()
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if not (model_cls := getattr(tv_models, config.type, None)):
        raise ValueError(f"Invalid model name: {config.type}")
    backbone_instance = model_cls(pretrained=config.pretrained)

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
