"""Factory module for instantiation of models.

Immutable Configs:
------------------
Please note that configuration objects may be mutated during the model creation
process (e.g., adjusting head_config's first layer in_channels based on dynamic
channel computation). If you need to reuse configuration objects or expect them
to remain unchanged, consider using deep copies. Future versions may enforce
immutability through explicit validation.

Error Handling & Validation:
----------------------------
Future improvements should include validation checks after head_config
adjustments. For example, verifying that the first layer's in_channels matches
the computed value and that the last layer's out_channels is consistent with the
model's expected output channels. This would help prevent configuration errors
and ensure consistent behavior across FusionModel and SegmentationModel.
"""

import torchvision.models as tv_models

from config.config_schema import BackboneModelConfig, FusionModelConfig, SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.segmentation import SegmentationModel
from models.utils import (
    adjust_head_config,
    freeze_model,
    headless,
    model_out_channels,
    model_weights,
)


def _create_backbone_model(config: BackboneModelConfig) -> BackboneModel:
    """Return BackboneModel instance based on configuration."""
    if not (model_cls := getattr(tv_models, config.type.lower(), None)):
        raise ValueError(f"Invalid model name: {config.type}")

    weights = model_weights(model_name=config.type) if config.pretrained else None
    backbone_instance = model_cls(weights=weights)
    if config.remove_head:
        backbone_instance = headless(backbone_instance)
    if config.freeze_backbone:
        freeze_model(backbone_instance)

    backbone_out_channels = model_out_channels(backbone_instance)
    config.head_config = adjust_head_config(config.head_config, backbone_out_channels)

    return BackboneModel(model=backbone_instance, config=config)


def create_segmentation_model(
    backbone_2d_config: BackboneModelConfig,
    backbone_3d_config: BackboneModelConfig,
    fusion_config: FusionModelConfig,
    segmentation_config: SegmentationModelConfig,
) -> SegmentationModel:
    """Return SegmentationModel instance based on configuration."""

    backbone2d_model = _create_backbone_model(backbone_2d_config)
    backbone3d_model = _create_backbone_model(backbone_3d_config)

    fusion_in_channels = backbone2d_model.out_channels + backbone3d_model.out_channels
    fusion_config.head_config = adjust_head_config(fusion_config.head_config, fusion_in_channels)
    fusion_model = FusionModel(fusion_config)

    segmentation_in_channels = fusion_model.out_channels
    segmentation_config.head_config = adjust_head_config(
        segmentation_config.head_config, segmentation_in_channels
    )

    return SegmentationModel(segmentation_config, backbone2d_model, backbone3d_model, fusion_model)
