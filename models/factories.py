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

from torch import nn

from config.config_schema import BackboneConfig, DataConfig, LayerConfig
from models.feature_extractor import FeatureExtractor, MultiBranchFeatureExtractor
from models.predictor import Predictor
from models.utils import (
    DTypeConverter,
    config_out_channels,
    freeze_model,
    get_tv_model,
    model_dtype,
    model_out_channels,
    remove_head,
    update_in_channels,
)


def _single_layer_module(config: LayerConfig) -> nn.Module:
    """Return a (single layer) module based on configuration."""
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


def _multi_layer_module(configs: List[LayerConfig]) -> nn.Module:
    """Return a (multi layer) module based on configuration, first layer is DTypeConverter."""
    if len(configs) == 0:
        return nn.Identity()
    modules_list: List[nn.Module] = [_single_layer_module(config) for config in configs]
    if (dtype := model_dtype(modules_list[0])) is not None:
        modules_list.insert(0, DTypeConverter(dtype))
    return nn.Sequential(*modules_list)


def _build_module(configs: List[LayerConfig], previous_out_channels: int) -> Tuple[nn.Module, int]:
    """Return a model with optional dtype conversion.

    NOTE:
    - This function is responsible for updating in_channels and computing out_channels.
    - The first layer of the module is a DTypeConverter if the dtype of the model is not None.
    """
    configs = update_in_channels(configs, previous_out_channels)
    out_channels = config_out_channels(configs, previous_out_channels)
    return _multi_layer_module(configs), out_channels


def _build_backbone(config: Optional[BackboneConfig], out_channels: int) -> Tuple[nn.Module, int]:
    """Return BackboneModel instance based on configuration."""
    backbone_module = nn.Sequential()
    if config is None:
        return backbone_module, out_channels

    model = get_tv_model(config.model_name, config.sub_module, config.pretrained)
    if config.remove_head:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            print("WARNING: Model does not have a backbone attribute, removing head instead.")
            backbone = remove_head(model)
        model = backbone
    if config.freeze_backbone:
        freeze_model(model)

    backbone_module.append(model)
    if (dtype := model_dtype(model)) is not None:
        backbone_module.insert(0, DTypeConverter(dtype))
    return backbone_module, model_out_channels(model)


class BranchContainer(TypedDict, total=False):
    """Intermediate container for branches."""

    backbone: nn.Module
    head: nn.Module
    out_channels: int


def _build_branch(config_branch: dict, out_channels: int) -> BranchContainer:
    branch = BranchContainer()
    branch["backbone"], out_channels = _build_backbone(config_branch["backbone"], out_channels)
    branch["head"], out_channels = _build_module(config_branch["head_layers"], out_channels)
    branch["out_channels"] = out_channels
    return branch


def build_predictor_model(config: dict, config_data: DataConfig) -> Predictor:
    """
    1: Predictor
    1.1: instantiate MultiBranchFeatureExtractor
    1.1.1: instantiate branches FeatureExtractor
    1.1.1.1: per branch: construct the branch backbone
    1.1.1.2: per branch: construct the branch head
    1.1.1.3: per branch: instantiate FeatureExtractor for each
    1.1.2: construct fusion head
    1.1.3: construct MultiBranchFeatureExtractor
    1.2: construct predictor head
    1.3: instantiate Predictor
    """
    # instantiate branches FeatureExtractor
    branches: Dict[str, BranchContainer] = {
        branch_name: _build_branch(config_branch, out_channels=config_data.channels[branch_name])
        for branch_name, config_branch in config["models"]["branches"].items()
    }
    # branches' output will be concatenated in MultiBranchFeatureExtractor. The
    # concatenated output will be the input for the fusion head or predictor
    # head (if no fusion head). Either way, whoever the next model is, its
    # in_channels will be the sum of the out_channels of all branches
    out_channels = sum(branch["out_channels"] for branch in branches.values())
    feature_extractors = [
        FeatureExtractor(backbone=branch["backbone"], head=branch["head"])
        for branch in branches.values()
    ]

    # build MultiBranchFeatureExtractor
    fusion_head, out_channels = _build_module(
        config["models"]["fusion"]["head_layers"], out_channels
    )
    multi_branch_feature_extractor = MultiBranchFeatureExtractor(
        branches=feature_extractors,
        head=fusion_head,
        head_dtype=model_dtype(fusion_head),
    )

    # build Predictor
    predictor_head, out_channels = _build_module(
        config["models"]["predictor"]["head_layers"], out_channels
    )
    predictor_model = Predictor(multi_branch_feature_extractor, predictor_head)
    return predictor_model
