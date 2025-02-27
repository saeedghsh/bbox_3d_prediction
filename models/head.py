"""
Head Configuration Design Notes
--------------------------------

This module implements a generic mechanism for building “head” modules from
configuration files. The intent is to support FusionModel and SegmentationModel
(and potentially others) by allowing the head to be defined entirely in a YAML
config.

Key Design Decisions:
---------------------
1. Dynamic Channel Assignment: - The first layer in the head config should have
   its 'in_channels' set to null (None in Python).
     This signals that the value must be determined dynamically at runtime based
     on the output channels of the preceding module (e.g., the fusion input
     channels computed from the 2D and 3D backbones).
   - It is the responsibility of the caller (e.g. FusionModel or
     SegmentationModel) to override this null value with the appropriate
     computed channel count.

2. Output Channel Consistency: - The last layer's 'out_channels' in the head
   configuration should match the desired output channels
     of the overall model (fusion or segmentation). Users must ensure
     consistency manually in the YAML config.
   - This approach minimizes over-engineering and assumes that users setting up
     the config are aware of the necessary channel sizes.

3. Generic Configuration: - The head configuration is defined generically using
   a list of layer definitions (each with a 'type'
     and arbitrary kwargs). This enables flexibility for various layer types
     without hard-coding logic.
   - Currently, no automatic propagation of channel sizes is performed between
     layers. Future improvements could add validation or automated inference of
     intermediate channel sizes.

4. Future Enhancements: - Optional validation checks can be added to verify that
   the first layer's in_channels is properly replaced
     at runtime and that the final layer's out_channels matches the expected
     value.
   - Additional layer types and more complex configurations may be supported by
     extending the `layers_map` in the build_head function.

Usage Guidance:
---------------
- Place your head configuration under the appropriate model section in your YAML
  config file. For example, for FusionModel:

      model:
        fusion:
          out_channels: 128 head_config:
            layers:
              - type: "Conv2d" in_channels: null out_channels: 128 kernel_size:
                3 stride: 1 padding: 1
              - type: "ReLU"

- Ensure that you manually set the out_channels for layers where necessary, and
  leave the first layer's in_channels as null so that the model code can
  dynamically set it.
- This module is designed to keep head-building logic decoupled from
  model-specific implementations. Similar mechanisms are used for BackboneModel,
  FusionModel, and SegmentationModel to maintain consistency.

"""

from copy import deepcopy
from typing import cast

from torch import nn

from config.config_schema import HeadConfig, LayerConfig


def _build_layer(config: LayerConfig) -> nn.Module:
    # Extend this map for other layer types as needed
    try:
        layer_cls = getattr(nn, config.layer_type)
    except AttributeError as e:
        raise ValueError(f"Unsupported layer type: {config.layer_type}") from e
    try:
        layer_instance = layer_cls(**config.kwargs)
    except Exception as e:
        raise ValueError(
            f"Error constructing layer {config.layer_type} with args {config.kwargs}: {e}"
        ) from e
    return cast(nn.Module, layer_instance)


def build_head(config: HeadConfig) -> nn.Module:
    """Return a head module based on configuration."""
    layers = [_build_layer(layer_cfg) for layer_cfg in config["layers"]]
    return nn.Sequential(*layers)


def adjust_head_config(config: HeadConfig, in_channels: int) -> HeadConfig:
    """
    Adjusts the head configuration by setting the 'in_channels' of the first layer
    to the provided value if it is None.

    Args:
        head_config: The original head configuration dictionary.
        in_channels: The computed input channels from the preceding module.

    Returns:
        A new head configuration with the first layer's in_channels set if needed.
    """
    adjusted_config = deepcopy(config)
    layers = adjusted_config.get("layers", [])
    if layers and layers[0].kwargs.get("in_channels") is None:
        layers[0].kwargs["in_channels"] = in_channels
    return adjusted_config
