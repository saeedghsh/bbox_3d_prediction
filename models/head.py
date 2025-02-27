"""Utility functions for building the head of a model."""

from torch import nn

from config.config_schema import HeadConfig, LayerConfig


def _build_layer(config: LayerConfig) -> nn.Module:
    # Extend this map for other layer types as needed
    layers_map = {"conv1x1": nn.Conv2d}
    try:
        layer_cls = layers_map[config.layer_type]
    except KeyError as e:
        raise ValueError(f"Unsupported layer type: {config.layer_type}") from e
    try:
        layer_instance = layer_cls(**config.kwargs)
    except Exception as e:
        raise ValueError(
            f"Error constructing layer {config.layer_type} with args {config.kwargs}: {e}"
        ) from e
    return layer_instance


def build_head(config: HeadConfig) -> nn.Module:
    """Return a head module based on configuration."""
    layers = [_build_layer(layer_cfg) for layer_cfg in config["layers"]]
    head = nn.Sequential(*layers)
    return head
