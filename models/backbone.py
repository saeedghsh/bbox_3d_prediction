"""Backbone Model interface for 2D and 3D branches."""

from typing import cast

from torch import Tensor, nn

from config.config_schema import BackboneModelConfig
from models.utils import build_head, dtype_matcher, model_out_channels


class BackboneModel(nn.Module):
    # pylint: disable=missing-function-docstring
    """Generic backbone model wrapper for 2D and 3D branches.
    Extracts spatial features from an input image/point-cloud using a pretrained
    model. Optionally removes the original head and attaches a new one dynamically.
    """

    def __init__(self, model: nn.Module, config: BackboneModelConfig) -> None:
        super().__init__()
        self._config = config
        self._backbone = model
        self._head = build_head(config.head_config)
        self._out_channels = model_out_channels(self._head)
        self._dtype_matchers = {"backbone_to_head": dtype_matcher(self._backbone, self._head)}

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head(self) -> nn.Module:
        return self._head

    def forward(self, x: Tensor) -> Tensor:
        features = self._backbone(x)
        features = self._dtype_matchers["backbone_to_head"](features)
        features = self._head(features)
        return cast(Tensor, features)
