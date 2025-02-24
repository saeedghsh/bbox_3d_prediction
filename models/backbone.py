"""Backbone Model interfaces for 2D and 3D branches."""

from torch import Tensor, nn


class Backbone2DModel(nn.Module):
    """Generic 2D backbone model wrapper.

    Extracts spatial features from an input image using a pretrained model.
    A 1x1 convolution adjusts the output channel dimension.
    """

    def __init__(self, backbone: nn.Module, in_channels: int, out_features: int) -> None:
        super().__init__()
        self._backbone = backbone
        self._in_channels = in_channels
        self._out_features = out_features
        self._conv = nn.Conv2d(in_channels, out_features, kernel_size=1)

    @property
    def backbone(self) -> nn.Module:
        """Returns the 2D backbone model."""
        return self._backbone

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self._backbone(x)
        out: Tensor = self._conv(features)
        return out


class Backbone3DModel(nn.Module):
    """Generic 3D backbone model wrapper for ordered point clouds.

    Expects a pretrained backbone (e.g. a Swin-T model) that outputs a spatial feature map.
    A 1x1 convolution adjusts the channel dimension.
    """

    def __init__(self, backbone: nn.Module, in_channels: int, out_features: int) -> None:
        super().__init__()
        self._backbone = backbone
        self._in_channels = in_channels
        self._out_features = out_features
        self._conv = nn.Conv2d(in_channels, out_features, kernel_size=1)

    @property
    def backbone(self) -> nn.Module:
        """Returns the 3D backbone model."""
        return self._backbone

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features: Tensor = self._backbone(x)
        out: Tensor = self._conv(features)
        return out
