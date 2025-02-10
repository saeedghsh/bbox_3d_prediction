"""Model for fusion of 2D and 3D data."""

import torch
from torch import Tensor, nn

from models.model_2d import Pretrained2DModel
from models.model_3d import Simple3DModel


class FusionModel(nn.Module):  # type: ignore[misc]
    # pylint: disable=missing-class-docstring, missing-function-docstring
    def __init__(self, fusion_out: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self._model2d = Pretrained2DModel(out_features=128)
        self._model3d = Simple3DModel(out_features=128)
        self._fusion = nn.Sequential(
            nn.Linear(256, fusion_out), nn.ReLU(), nn.Linear(fusion_out, num_classes)
        )

    @property
    def model2d(self) -> Pretrained2DModel:
        return self._model2d

    @property
    def model3d(self) -> Simple3DModel:
        return self._model3d

    @property
    def fusion(self) -> nn.Sequential:
        return self._fusion

    def forward(self, image: Tensor, point_cloud: Tensor) -> Tensor:
        feat2d = self.model2d(image)
        feat3d = self.model3d(point_cloud)
        fused = torch.cat([feat2d, feat3d], dim=1)
        out: Tensor = self.fusion(fused)
        return out
