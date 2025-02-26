# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor, nn

from config.config_schema import BackboneModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.segmentation import SegmentationModel


class DummyBackbone(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self._conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: Tensor = self._conv(x)
        return out


@pytest.fixture
def backbone_fixture() -> DummyBackbone:
    return DummyBackbone(out_channels=32)


@pytest.fixture
def backbone_model_fixture(request: FixtureRequest) -> BackboneModel:
    backbone = request.getfixturevalue("backbone_fixture")
    config = BackboneModelConfig(
        model_name="DummyBackbone", in_channels=32, out_features=16, pretrained=False
    )
    return BackboneModel(model=backbone, config=config)


@pytest.fixture
def fusion_model_fixture() -> FusionModel:
    return FusionModel(feat2d_channels=16, feat3d_channels=16, out_channels=32)


@pytest.fixture
def segmentation_model_fixture(request: FixtureRequest) -> SegmentationModel:
    model2d = request.getfixturevalue("backbone_model_fixture")
    model3d = request.getfixturevalue("backbone_model_fixture")
    fusion = request.getfixturevalue("fusion_model_fixture")
    return SegmentationModel(
        backbone2d=model2d,
        backbone3d=model3d,
        fusion=fusion,
        out_channels=8,
    )


def test_backbone2d_model(request: FixtureRequest) -> None:
    model2d = request.getfixturevalue("backbone_model_fixture")
    input_tensor = torch.randn(2, 3, 64, 64)
    output_tensor = model2d(input_tensor)
    assert output_tensor.shape == (2, 16, 64, 64)


def test_backbone3d_model(request: FixtureRequest) -> None:
    model3d = request.getfixturevalue("backbone_model_fixture")
    input_tensor = torch.randn(2, 3, 64, 64)
    output_tensor = model3d(input_tensor)
    assert output_tensor.shape == (2, 16, 64, 64)


def test_fusion_model(request: FixtureRequest) -> None:
    fusion = request.getfixturevalue("fusion_model_fixture")
    feat2d = torch.randn(2, 16, 32, 32)
    feat3d = torch.randn(2, 16, 32, 32)
    output_tensor = fusion(feat2d, feat3d)
    assert output_tensor.shape == (2, 32, 32, 32)


def test_segmentation_model(request: FixtureRequest) -> None:
    seg_model = request.getfixturevalue("segmentation_model_fixture")
    image_in = torch.randn(2, 3, 64, 64)
    pointcloud_in = torch.randn(2, 3, 64, 64)
    output_tensor = seg_model(image_in, pointcloud_in)
    assert output_tensor.shape == (2, 8, 64, 64)
