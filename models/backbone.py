"""Backbone Model interface for 2D and 3D branches."""

from typing import Callable, cast

from torch import Tensor, nn

from config.config_schema import BackboneModelConfig
from models.head import build_head


def _headless(model: nn.Module) -> nn.Module:
    return cast(nn.Module, nn.Sequential(*list(model.children())[:-1]))


def _freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _dtype_matcher(backbone: nn.Module, head: nn.Module) -> Callable[[Tensor], Tensor]:
    backbone_out_dtype = next(backbone.parameters()).dtype if list(backbone.parameters()) else None
    head_params = list(head.parameters())
    head_in_dtype = head_params[0].dtype if head_params else None
    if backbone_out_dtype is None or head_in_dtype is None or backbone_out_dtype == head_in_dtype:
        return lambda x: x
    return lambda x: x.to(head_in_dtype)


class BackboneModel(nn.Module):
    """Generic backbone model wrapper for 2D and 3D branches.
    Extracts spatial features from an input image/point-cloud using a pretrained
    model. Optionally removes the original head and attaches a new one dynamically.
    """

    def __init__(self, model: nn.Module, config: BackboneModelConfig) -> None:
        super().__init__()
        self._config = config
        self._backbone = _headless(model) if config.remove_head else model
        if config.freeze_backbone:
            _freeze_model(self._backbone)
        self._head = build_head(config.head_config)
        self._match_dtype = _dtype_matcher(self._backbone, self._head)

    @property
    def backbone(self) -> nn.Module:  # pylint: disable=missing-function-docstring
        return self._backbone

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=missing-function-docstring
        features = self._match_dtype(self._backbone(x))
        return cast(Tensor, self._head(features))
