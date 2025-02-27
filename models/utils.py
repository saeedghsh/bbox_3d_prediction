"""This module contains utility functions for models."""

from typing import Callable, cast

from torch import Tensor, nn


def headless(model: nn.Module) -> nn.Module:
    """Returns a model without the last layer."""
    return cast(nn.Module, nn.Sequential(*list(model.children())[:-1]))


def freeze_model(model: nn.Module) -> None:
    """Freezes all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def dtype_matcher(m1: nn.Module, m2: nn.Module) -> Callable[[Tensor], Tensor]:
    """Returns a conversion function from m1 out dtype to m2 in dtype."""
    m1_out_dtype = next(m1.parameters()).dtype if list(m1.parameters()) else None
    m2_params = list(m2.parameters())
    m2_in_dtype = m2_params[0].dtype if m2_params else None
    if m1_out_dtype is None or m2_in_dtype is None or m1_out_dtype == m2_in_dtype:
        return lambda x: x
    return lambda x: x.to(m2_in_dtype)
