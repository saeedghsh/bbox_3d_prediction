"""This module contains utility functions for models."""

from typing import Callable, Optional, cast

from torch import Tensor, dtype, nn


def headless(model: nn.Module) -> nn.Module:
    """Returns a model without the last layer."""
    return cast(nn.Module, nn.Sequential(*list(model.children())[:-1]))


def freeze_model(model: nn.Module) -> None:
    """Freezes all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def model_dtype(m: nn.Module) -> Optional[dtype]:
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

    raise ValueError("Could not determine out_channels. Model might be unconventional.")
