"""Feature extraction models."""

from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from models.utils import DTypeConverter, TensorConcatenator


class FeatureExtractor(nn.Module):
    """A model that extracts features from an input tensor with optional dtype conversion."""

    def __init__(
        self,
        backbone: Optional[nn.Module],
        backbone_dtype: Optional[torch.dtype],
        head: Optional[nn.Module],
        head_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        backbone_layers: List[nn.Module] = []
        if backbone is not None:
            if backbone_dtype is not None:
                backbone_layers.append(DTypeConverter(backbone_dtype))
            backbone_layers.append(backbone)

        head_layers: List[nn.Module] = []
        if head is not None:
            if head_dtype is not None:
                head_layers.append(DTypeConverter(head_dtype))
            head_layers.append(head)

        self._model = (
            nn.Sequential(*backbone_layers, *head_layers)
            if backbone_layers or head_layers
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self._model(x))


class MultiBranchFeatureExtractor(nn.Module):
    """Processes multi-branch inputs, fuses features, and applies a head if provided."""

    def __init__(
        self,
        branches: List[FeatureExtractor],
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
