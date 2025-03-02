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
from torch import nn

from config.config_schema import BackboneConfig, LayerConfig
from models.feature_extractor import MultiBranchFeatureExtractor, StackedModel
from models.predictor import Predictor
from models.utils import (
    freeze_model,
    get_tv_model,
    headless,
    model_dtype,
    model_out_channels,
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


def build_backbone(
    config: Optional[BackboneConfig], out_channels: int
) -> Tuple[Optional[nn.Module], Optional[torch.dtype], int]:
    """Return BackboneModel instance based on configuration."""
    if config is None:
        return None, None, out_channels
    model = get_tv_model(config.model_name, config.sub_module, config.pretrained)
    if config.remove_head:
        model = headless(model)
    if config.freeze_backbone:
        freeze_model(model)
    return model, model_dtype(model), model_out_channels(model)


def _build_branch(config_branch: dict, out_channels: int) -> BranchContainer:
    branch = BranchContainer()
    branch["backbone"], branch["backbone_dtype"], out_channels = build_backbone(
        config_branch["backbone"], out_channels
    )
    branch["head"], branch["head_dtype"], out_channels = _build_head(
        config_branch["head_layers"], out_channels
    )
    branch["out_channels"] = out_channels
    return branch


def _sum_branch_out_channels(branches: Dict[str, BranchContainer]) -> int:
    # branches output will be concatenated in MultiBranchFeatureExtractor. The
    # concatenated output will be the input for the fusion head or predictor
    # head (if no fusion head). Either way, whoever the next model is, its
    # in_channels will be the sum of the out_channels of all branches
    return sum(branch["out_channels"] for branch in branches.values())


def _build_feature_extractors(
    config_branches: dict, config_data: dict
) -> Tuple[List[StackedModel], int]:
    branches: Dict[str, BranchContainer] = {}
    for branch_name, config_branch in config_branches.items():
        out_channels = config_data["channels"][branch_name]
        branches[branch_name] = _build_branch(config_branch, out_channels)
    feature_extractors = [
        StackedModel(
            models=[
                (branch["backbone"], branch["backbone_dtype"]),
                (branch["head"], branch["head_dtype"]),
            ]
        )
        for branch in branches.values()
    ]
    out_channels = _sum_branch_out_channels(branches)
    return feature_extractors, out_channels


def build_predictor_model(config: dict) -> Predictor:
    """
    1: Predictor
    1.1: instantiate MultiBranchFeatureExtractor
    1.1.1: instantiate branches StackedModel
    1.1.1.1: per branch: construct the branch backbone
    1.1.1.2: per branch: construct the branch head
    1.1.1.3: per branch: instantiate StackedModel for each
    1.1.2: construct fusion head
    1.1.3: construct MultiBranchFeatureExtractor
    1.2: construct predictor head
    1.3: instantiate Predictor
    """
    # build List[StackedModel]
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
    # build Predictor
    predictor_head, predictor_head_dtype, out_channels = _build_head(
        config=config["models"]["predictor"]["head_layers"],
        out_channels=out_channels,
    )
    predictor_model = Predictor(
        multi_branch_feature_extractor=multi_branch_feature_extractor,
        head=predictor_head,
        head_dtype=predictor_head_dtype,
    )
    return predictor_model
