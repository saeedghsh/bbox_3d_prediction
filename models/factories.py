"""Factory module for instantiation of models.

Immutable Configs:
------------------
Please note that configuration objects may be mutated during the model creation
process (e.g., adjusting head config's first layer in_channels based on dynamic
channel computation). If you need to reuse configuration objects or expect them
to remain unchanged, consider using deep copies. Future versions may enforce
immutability through explicit validation.

Error Handling & Validation:
----------------------------
Future improvements should include validation checks after head config layers'
adjustments.
"""

from typing import Dict, List, Optional, Tuple, TypedDict, cast

import torch
import torchvision.models as tv_models
from torch import nn

from config.config_schema import BackboneConfig, LayerConfig
from models.feature_extractor import FeatureExtractor, MultiBranchFeatureExtractor
from models.segmentation import SegmentationModel
from models.utils import (
    freeze_model,
    headless,
    model_dtype,
    model_out_channels,
    model_weights,
    set_in_channels,
)


class BranchContainer(TypedDict, total=False):
    """Intermediate container for branches."""

    backbone: Optional[nn.Module]
    backbone_dtype: Optional[torch.dtype]
    head: Optional[nn.Module]
    head_dtype: Optional[torch.dtype]
    out_channels: int


def _build_layer(config: LayerConfig) -> nn.Module:
    try:
        layer_cls = getattr(nn, config.type)
    except AttributeError as e:
        raise ValueError(f"Unsupported layer type: {config.type}") from e
    try:
        layer_instance = layer_cls(**config.kwargs)
    except Exception as e:
        raise ValueError(
            f"Error constructing layer {config.type} with args {config.kwargs}: {e}"
        ) from e
    return cast(nn.Module, layer_instance)


def _build_layers(config: List[LayerConfig]) -> nn.Module:
    layers = [_build_layer(layer_config) for layer_config in config]
    return nn.Sequential(*layers)


def _build_head(
    config: List[LayerConfig], out_channels: int
) -> Tuple[Optional[nn.Module], Optional[torch.dtype], int]:
    """Return a head module based on configuration."""
    if len(config) == 0:
        return None, None, out_channels
    config, out_channels = set_in_channels(config, out_channels)
    head = _build_layers(config)
    dtype = model_dtype(head)
    return head, dtype, out_channels


def _build_backbone(
    config: Optional[BackboneConfig], out_channels: int
) -> Tuple[Optional[nn.Module], Optional[torch.dtype], int]:
    """Return BackboneModel instance based on configuration."""
    if config is None:
        return None, None, out_channels

    if not (model_cls := getattr(tv_models, config.model_name.lower(), None)):
        raise ValueError(f"Invalid model name: {config.model_name}")

    weights = model_weights(config.model_name) if config.pretrained else None
    model: nn.Module = model_cls(weights=weights)
    if config.remove_head:
        model = headless(model)
    if config.freeze_backbone:
        freeze_model(model)
    return model, model_dtype(model), model_out_channels(model)


def _build_branch(config_branch: dict, out_channels: int) -> BranchContainer:
    branch = BranchContainer()
    branch["backbone"], branch["backbone_dtype"], out_channels = _build_backbone(
        config_branch["backbone"], out_channels
    )
    branch["head"], branch["head_dtype"], out_channels = _build_head(
        config_branch["head_layers"], out_channels
    )
    branch["out_channels"] = out_channels
    return branch


def _sum_branch_out_channels(branches: Dict[str, BranchContainer]) -> int:
    # branches output will be concatenated in MultiBranchFeatureExtractor. The
    # concatenated output will be the input for the fusion head or segmentation
    # head (if no fusion head). Either way, whoever the next model is, its
    # in_channels will be the sum of the out_channels of all branches
    return sum(branch["out_channels"] for branch in branches.values())


def _build_feature_extractors(
    config_branches: dict, config_data: dict
) -> Tuple[List[FeatureExtractor], int]:
    branches: Dict[str, BranchContainer] = {}
    for branch_name, config_branch in config_branches.items():
        out_channels = config_data["channels"][branch_name]
        branches[branch_name] = _build_branch(config_branch, out_channels)
    feature_extractors = [
        FeatureExtractor(
            backbone=branch["backbone"],
            backbone_dtype=branch["backbone_dtype"],
            head=branch["head"],
            head_dtype=branch["head_dtype"],
        )
        for branch in branches.values()
    ]
    out_channels = _sum_branch_out_channels(branches)
    return feature_extractors, out_channels


def build_segmentation_model(config: dict) -> SegmentationModel:
    """
    1: SegmentationModel
    1.1: instantiate MultiBranchFeatureExtractor
    1.1.1: instantiate branches FeatureExtractor
    1.1.1.1: per branch: construct the branch backbone
    1.1.1.2: per branch: construct the branch head
    1.1.1.3: per branch: instantiate FeatureExtractor for each
    1.1.2: construct fusion head
    1.1.3: construct MultiBranchFeatureExtractor
    1.2: construct segmentation head
    1.3: instantiate SegmentationModel
    """

    # build List[FeatureExtractor]
    feature_extractors, out_channels = _build_feature_extractors(
        config_branches=config["models"]["branches"],
        config_data=config["data"],
    )

    # build MultiBranchFeatureExtractor
    fusion_head, fusion_head_dtype, out_channels = _build_head(
        config=config["models"]["fusion"]["head_layers"],
        out_channels=out_channels,
    )
    multi_branch_feature_extractor = MultiBranchFeatureExtractor(
        branches=feature_extractors,
        head=fusion_head,
        head_dtype=fusion_head_dtype,
    )

    # build SegmentationModel
    segmentation_head, segmentation_head_dtype, out_channels = _build_head(
        config=config["models"]["segmentation"]["head_layers"],
        out_channels=out_channels,
    )
    segmentation_model = SegmentationModel(
        multi_branch_feature_extractor=multi_branch_feature_extractor,
        head=segmentation_head,
        head_dtype=segmentation_head_dtype,
    )
    return segmentation_model
