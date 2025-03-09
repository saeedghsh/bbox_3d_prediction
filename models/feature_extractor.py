"""Feature extraction models."""

from typing import Dict, Optional, cast

import torch
from torch import Tensor, nn

from models.utils import DTypeConverter, TensorConcatenator


class FeatureExtractor(nn.Module):
    """A model that extracts features from an input tensor with optional dtype conversion."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self._backbone = backbone
        self._head = head

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self._backbone(x)
        if isinstance(x, dict):  # handling output of backbone if returns a dict
            x = x["0"]
        x = self._head(x)
        return x


class MultiBranchFeatureExtractor(nn.Module):
    """Processes multi-branch inputs, fuses features, and applies a head if provided."""

    def __init__(
        self,
        branches: Dict[str, FeatureExtractor],
        head: nn.Module,
        head_dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self._branches = nn.ModuleDict(branches)
        dtype_convertor = DTypeConverter(head_dtype) if head_dtype is not None else None
        self._concatenator = TensorConcatenator(dtype_convertor=dtype_convertor)
        self._head = head

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        """Forward pass through branches, concatenation, and head."""
        if x.keys() != self._branches.keys():
            raise ValueError(
                f"branch keys ({self._branches.keys()}) do not match input: {x.keys()}"
            )
        x_list = [branch(x[branch_name]) for branch_name, branch in self._branches.items()]
        x_list = self._concatenator(x_list)
        return cast(Tensor, self._head(x_list))
