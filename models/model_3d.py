"""Model for 3D data."""

from torch import Tensor, nn


class Pretrained3DModel(nn.Module):  # type: ignore[misc]
    # pylint: disable=missing-class-docstring, missing-function-docstring
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self._fc = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, out_features))

    @property
    def fc(self) -> nn.Sequential:
        return self._fc

    def forward(self, point_cloud: Tensor) -> Tensor:
        # Assuming point_cloud is (B, N, 3)
        # Simple aggregation: mean pooling over points
        features = self.fc(point_cloud)  # (B, N, out_features)
        out: Tensor = features.mean(dim=1)  # (B, out_features)
        return out
