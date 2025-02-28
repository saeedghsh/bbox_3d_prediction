"""Backbone Model interface for 2D and 3D branches."""

from typing import cast

from torch import Tensor, nn

from config.config_schema import BackboneModelConfig
from models.head import build_head
from models.utils import dtype_matcher, freeze_model, headless


class BackboneModel(nn.Module):
    # pylint: disable=missing-function-docstring
    """Generic backbone model wrapper for 2D and 3D branches.
    Extracts spatial features from an input image/point-cloud using a pretrained
    model. Optionally removes the original head and attaches a new one dynamically.
    """

    def __init__(self, model: nn.Module, config: BackboneModelConfig) -> None:
        super().__init__()
        self._config = config
        self._backbone = headless(model) if config.remove_head else model
        if config.freeze_backbone:
            freeze_model(self._backbone)
        self._head = build_head(config.head_config)
        self._dtype_matchers = {"backbone_to_head": dtype_matcher(self._backbone, self._head)}

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
