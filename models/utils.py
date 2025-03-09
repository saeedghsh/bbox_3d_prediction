"""This module contains utility functions for models.


Head Configuration Design Notes
--------------------------------

This module implements a generic mechanism for building “head” modules from
configuration files. The intent is to support any Model with a head by allowing
the head to be defined entirely in a YAML config.

Key Design Decisions:
---------------------
1. Dynamic Channel Assignment: No first layer in the head config should have its
   'in_channels' set. This signals that the value must be determined dynamically
   at runtime based on the output channels of the preceding module channels,
   starting from data channels.

2. Output Channel Consistency: The 'out_channels' of the last layer of the last
   model in inference pipeline should match the desired output channels. Users
   must ensure consistency manually in the YAML config, as part of model
   architecture design. This approach minimizes over-engineering and assumes
   that users setting up the config are aware of the necessary channel sizes.

3. Generic Configuration: The head configuration is defined generically using a
   list of layer definitions (each with a 'type' and arbitrary kwargs). This
   enables flexibility for various layer types without hard-coding logic.
   Currently, an automatic propagation of channel sizes is performed between
   layers. Future improvements should add validation of intermediate channel
   sizes, and in and out channel sizes of the whole model.

Usage Guidance:
---------------
config file. For example, for fusion (used as head in
MultiBranchFeatureExtractor):

    model: fusion:
        head_layers:
            - type: "Conv2d"
              out_channels: 128
              kernel_size: 3
              stride: 1
              padding: 1
            - type: "ReLU"

- Ensure that you manually set the out_channels for layers as necessary.
- This module is designed to keep head-building logic decoupled from
  model-specific implementations.
"""

from types import ModuleType
from typing import Callable, List, Optional, cast

import torch
import torchvision.models as tv_models
from torch import Tensor, nn

from config.config_schema import LayerConfig


class DTypeConverter(nn.Module):
    """A layer that converts tensor dtype"""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.to(dtype=self.dtype)


class TensorConcatenator(nn.Module):
    """Concatenate feature maps from multiple branches with shape validation."""

    def __init__(self, dtype_convertor: Optional[nn.Module] = None) -> None:
        super().__init__()
        self._dtype_convertor = dtype_convertor or nn.Identity()

    def forward(self, tensors: List[Tensor]) -> Tensor:
        """Forward pass."""
        shapes = {t.shape[2:] for t in tensors}  # Extract spatial dimensions (H, W, ...)
        if len(shapes) > 1:
            raise ValueError(f"Inconsistent tensor shapes: {shapes}")
        return torch.cat(self._dtype_convertor(tensors), dim=1)


def _get_tv_models_sub_module(sub_module_name: str = "") -> ModuleType:
    """Return a torchvision sub-module by name, torchvision itself is name is empty."""
    sub_module = tv_models
    if sub_module_name:
        try:
            sub_module = getattr(tv_models, sub_module_name)
        except AttributeError as e:
            raise ValueError(f"Invalid sub_module name: {sub_module_name}: {e}") from e
    return cast(ModuleType, sub_module)


def get_tv_model(model_name: str, sub_module_name: str = "", pretrained: bool = True) -> nn.Module:
    """Return a torchvision model by name."""
    sub_module = _get_tv_models_sub_module(sub_module_name)

    try:
        model_cls = getattr(sub_module, model_name.lower())
    except AttributeError as e:
        raise ValueError(f"Invalid model name: {model_name.lower()}: {e}") from e

    weights = None
    if pretrained:
        try:
            weights_enum = getattr(sub_module, f"{model_name}_Weights")
            weights = weights_enum.DEFAULT
        except AttributeError as e:
            raise ValueError(f"Could not load weights for model: {model_name}: {e}") from e

    return cast(nn.Module, model_cls(weights=weights))


def remove_head(model: nn.Module, layer_counts: int = 1) -> nn.Module:
    """Returns a model without the last layer(s)."""
    return cast(nn.Module, nn.Sequential(*list(model.children())[:-layer_counts]))


def freeze_model(model: nn.Module) -> None:
    """Freezes all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def model_dtype(m: nn.Module) -> Optional[torch.dtype]:
    """Returns the dtype of the first parameter of the model."""
    params = list(m.parameters())
    return params[0].dtype if params else None


def dtype_matcher(m1: nn.Module, m2: nn.Module) -> Callable[[Tensor], Tensor]:
    """Returns a conversion function from m1 out dtype to m2 in dtype.

    NOTE: assumes dtype in the model does not vary."""
    m1_dtype = model_dtype(m1)
    m2_dtype = model_dtype(m2)
    if m1_dtype is None or m2_dtype is None or m1_dtype == m2_dtype:
        return lambda x: x
    return lambda x: x.to(m2_dtype)


def model_out_channels(model: nn.Module) -> int:
    """Extracts the output channel size of a given PyTorch model.

    NOTE: Not guaranteed to be correct.
    """
    last_layer = None

    # Traverse layers in reverse to find the last defining layer
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
        if isinstance(layer, nn.Linear):
            return layer.out_features
        last_layer = layer  # Fallback to the last registered layer

    # Handle custom architectures (fallback to last known layer)
    out_channels = None
    if last_layer and hasattr(last_layer, "out_channels"):
        out_channels = getattr(last_layer, "out_channels")
    elif last_layer and hasattr(last_layer, "out_features"):
        out_channels = getattr(last_layer, "out_features")

    if out_channels and isinstance(out_channels, int):
        return out_channels  # type: ignore

    err_msg = "Could not determine out_channels. Model might be unconventional."
    raise ValueError(
        f"{err_msg}. Last layer type is: {last_layer.type}"  # type: ignore[union-attr]
    )


def _layers_with_out_channels() -> List[str]:
    """Return a list of layer types that accept out_channels as input arg."""
    return ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]


def update_in_channels(configs: List[LayerConfig], previous_out_channels: int) -> List[LayerConfig]:
    """Update (actually 'set') in_channels for each layer in the list."""
    out_channels = previous_out_channels
    for config in configs:
        if config.type not in _layers_with_out_channels():
            continue
        config.kwargs["in_channels"] = out_channels
        if not (out_channels := config.kwargs.get("out_channels")):  # type: ignore
            raise ValueError(f"out_channels must be defined for {config.type}")
    return configs


def config_out_channels(configs: List[LayerConfig], last_out_channels: int) -> int:
    """Compute the last out_channels from the list of layers."""
    out_channels = last_out_channels
    for config in configs:
        if config.type not in _layers_with_out_channels():
            continue
        if not (out_channels := config.kwargs.get("out_channels")):  # type: ignore
            raise ValueError(f"out_channels must be defined for {config.type}")
    return out_channels
