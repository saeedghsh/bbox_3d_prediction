"""Segmentation model that integrates 2D/3D backbones, fusion, and segmentation head."""

from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from models.utils import DTypeConverter


class SegmentationModel(nn.Module):
    """A model that predicts a target tensor from an input tensor."""

    def __init__(
        self,
        multi_branch_feature_extractor: nn.Module,
        head: Optional[nn.Module],
        head_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        head_layers: List[nn.Module] = []
        if head is not None:
            if head_dtype is not None:
                head_layers.append(DTypeConverter(head_dtype))
            head_layers.append(head)

        self._model = nn.Sequential(multi_branch_feature_extractor, *head_layers)

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self._model(x))
