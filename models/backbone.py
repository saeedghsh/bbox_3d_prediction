"""Backbone Model interface for 2D and 3D branches."""

from typing import cast

from torch import Tensor, nn

from config.config_schema import BackboneModelConfig


class BackboneModel(nn.Module):
    """Generic backbone model wrapper for 2D and 3D branches.

    Extracts spatial features from an input image/point-cloud using a pretrained
    model. A 1x1 convolution adjusts the output channel dimension.
    """

    def __init__(self, model: nn.Module, config: BackboneModelConfig) -> None:
        super().__init__()
        self._model = model
        self._conv = nn.Conv2d(config.in_channels, config.out_features, kernel_size=1)

    @property
    def backbone(self) -> nn.Module:
        """Returns the 2D backbone model."""
        return self._model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self._model(x)
        return cast(Tensor, self._conv(features))
