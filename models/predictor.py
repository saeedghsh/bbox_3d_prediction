"""Predictor model that integrates multi branch feature extractor, and a predictor head."""

from typing import List, cast

from torch import Tensor, nn


class Predictor(nn.Module):
    """A model that predicts a target tensor from an input tensor."""

    def __init__(self, multi_branch_feature_extractor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self._model = nn.Sequential(multi_branch_feature_extractor, head)

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self._model(x))
