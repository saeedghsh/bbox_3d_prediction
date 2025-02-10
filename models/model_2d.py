"""Model for 2D data."""

from torch import Tensor, nn
from torchvision import models


class Pretrained2DModel(nn.Module):  # type: ignore[misc]
    # pylint: disable=missing-class-docstring, missing-function-docstring
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the last FC layer
        self._features = nn.Sequential(*list(resnet.children())[:-1])
        self._fc = nn.Linear(resnet.fc.in_features, out_features)

    @property
    def features(self) -> nn.Sequential:
        return self._features

    @property
    def fc(self) -> nn.Linear:
        return self._fc

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out: Tensor = self.fc(x)
        return out
