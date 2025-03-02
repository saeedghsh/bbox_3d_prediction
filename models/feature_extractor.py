"""Feature extraction models."""

from typing import List, Optional, Tuple, cast

import torch
from torch import Tensor, nn

from models.utils import DTypeConverter, TensorConcatenator


def build_layers(model: Optional[nn.Module], model_dtype: Optional[torch.dtype]) -> List[nn.Module]:
    """Return a list with dtype converter followed by model.

    If either is None, the corresponding layer is not added."""
    layers: List[nn.Module] = []
    if model is not None:
        if model_dtype is not None:
            layers.append(DTypeConverter(model_dtype))
        layers.append(model)
    return layers


class StackedModel(nn.Module):
    """A model that extracts features from an input tensor with optional dtype conversion."""

    def __init__(self, models: List[Tuple[Optional[nn.Module], Optional[torch.dtype]]]) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            layer for model, dtype in models for layer in build_layers(model, dtype)
        ]
        self._model = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self._model(x))


class MultiBranchFeatureExtractor(nn.Module):
    """Processes multi-branch inputs, fuses features, and applies a head if provided."""

    def __init__(
        self,
        branches: List[StackedModel],
        head: Optional[nn.Module],
        head_dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self._branches = nn.ModuleList(branches)
        self._convertor = DTypeConverter(head_dtype) if head_dtype is not None else nn.Identity()
        self._concatenator = TensorConcatenator()
        self._head = head if head is not None else nn.Identity()

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward pass through branches, concatenation, and head."""
        if len(x) != len(self._branches):
            raise ValueError(
                f"number of branches ({len(self._branches)}) does not match input: {len(x)}"
            )
        x = [self._convertor(branch(x_i)) for branch, x_i in zip(self._branches, x)]
        x = self._concatenator(x)
        return cast(Tensor, self._head(x))
