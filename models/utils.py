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
