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

from typing import Dict, Optional

import torch
import torchvision.models as tv_models
from torchvision.models._api import WeightsEnum

from config.config_schema import BackboneModelConfig, FusionModelConfig, SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.head import adjust_head_config
from models.segmentation import SegmentationModel
from models.utils import model_dtype


def _backbone_weights(config: BackboneModelConfig) -> Optional[WeightsEnum]:
    weights = None
    if config.pretrained:
        weights_enum = getattr(tv_models, f"{config.type}_Weights", None)
        weights = weights_enum.DEFAULT if weights_enum else None
    if weights is None:
        raise ValueError(f"Could not load weights for model: {config.type}")
    return weights


def _create_backbone_model(config: BackboneModelConfig) -> BackboneModel:
    """Return BackboneModel instance based on configuration."""
    if not (model_cls := getattr(tv_models, config.type.lower(), None)):
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

    backbone2d_model = _create_backbone_model(backbone_2d_config)
    backbone3d_model = _create_backbone_model(backbone_3d_config)

    fusion_in_channels = backbone_2d_config.out_channels + backbone_3d_config.out_channels
    fusion_config.head_config = adjust_head_config(fusion_config.head_config, fusion_in_channels)
    fusion_model = FusionModel(fusion_config)

    segmentation_in_channels = fusion_config.out_channels
    segmentation_config.head_config = adjust_head_config(
        segmentation_config.head_config, segmentation_in_channels
    )

    return SegmentationModel(segmentation_config, backbone2d_model, backbone3d_model, fusion_model)


def input_dtypes(model: SegmentationModel) -> Dict[str, torch.dtype]:
    """Return expected input dtypes for 2D and 3D backbones."""
    dtype_2d = model_dtype(model.backbone2d.backbone)
    dtype_3d = model_dtype(model.backbone3d.backbone)
    if dtype_2d is None or dtype_3d is None:
        raise ValueError(f"Could not infer Backbone dtypes - 2D: {dtype_2d} - 3D {dtype_3d}")
    return {"2d": dtype_2d, "3d": dtype_3d}
