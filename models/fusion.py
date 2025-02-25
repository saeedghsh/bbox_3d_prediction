"""Model for fusion of 2D and 3D branches."""

import torch
from torch import Tensor, nn


class FusionModel(nn.Module):
    """Fusion head that fuses precomputed 2D and 3D feature maps.

    Expects as input two feature maps with the same spatial dimensions.
    """

    def __init__(self, feat2d_channels: int, feat3d_channels: int, out_channels: int) -> None:
        super().__init__()
        self._in_channels = feat2d_channels + feat3d_channels
        self._fusion_head = nn.Sequential(
            nn.Conv2d(self._in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()
        )
        self._out_channels = out_channels

    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self._out_channels

    def forward(self, feat2d: Tensor, feat3d: Tensor) -> Tensor:
        """Forward pass."""
        fused: Tensor = torch.cat([feat2d, feat3d], dim=1)
        out: Tensor = self._fusion_head(fused)
        return out
