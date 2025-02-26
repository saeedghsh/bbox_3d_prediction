"""Model for fusion of 2D and 3D branches."""

from typing import cast

import torch
from torch import Tensor, nn

from config.config_schema import BackboneModelConfig, FusionModelConfig


class FusionModel(nn.Module):
    """Fusion head that fuses precomputed 2D and 3D feature maps.

    Expects as input two feature maps with the same spatial dimensions.
    """

    def __init__(
        self,
        fusion_config: FusionModelConfig,
        backbone_2d_config: BackboneModelConfig,
        backbone_3d_config: BackboneModelConfig,
    ) -> None:
        super().__init__()
        self._config = fusion_config
        self._in_channels = backbone_2d_config.out_features + backbone_3d_config.out_features
        self._out_channels = fusion_config.out_channels

        self._fusion_head = nn.Sequential(
            nn.Conv2d(self._in_channels, fusion_config.out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    @property
    def config(self) -> FusionModelConfig:
        """Returns the model configuration."""
        return self._config

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self._config.out_channels

    def forward(self, feat2d: Tensor, feat3d: Tensor) -> Tensor:
        """Forward pass."""
        fused = torch.cat([feat2d, feat3d], dim=1)
        return cast(Tensor, self._fusion_head(fused))
