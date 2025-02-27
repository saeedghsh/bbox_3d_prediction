"""Model for fusion of 2D and 3D branches."""

from typing import List, cast

import torch
from torch import Tensor, nn

from config.config_schema import FusionModelConfig
from models.head import build_head


class FusionModel(nn.Module):
    # pylint: disable=missing-function-docstring
    """Fusion head that fuses multiple feature maps."""

    def __init__(self, config: FusionModelConfig) -> None:
        super().__init__()
        self._config = config
        self._head = build_head(self._config.head_config)

    @property
    def config(self) -> FusionModelConfig:
        return self._config

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def out_channels(self) -> int:
        return self._config.out_channels

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        fused = torch.cat(feature_maps, dim=1)
        return cast(Tensor, self._head(fused))
