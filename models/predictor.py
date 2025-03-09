"""Predictor model that integrates multi branch feature extractor, and a predictor head."""

from typing import Dict, cast

from torch import Tensor, nn


class Predictor(nn.Module):
    """A model that predicts a target tensor from an input tensor."""

    def __init__(self, multi_branch_feature_extractor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self._multi_branch_feature_extractor = multi_branch_feature_extractor
        self._head = head

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        """Forward pass."""
        x = self._multi_branch_feature_extractor(x)
        return cast(Tensor, self._head(x))
