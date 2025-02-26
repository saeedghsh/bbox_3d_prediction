# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring

from typing import cast

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor, nn

from config.config_schema import BackboneModelConfig, FusionModelConfig, SegmentationModelConfig
from models.backbone import BackboneModel
from models.fusion import FusionModel
from models.segmentation import SegmentationModel


class DummyModel(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self._conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(Tensor, self._conv(x))


@pytest.fixture
def model_fixture() -> DummyModel:
    return DummyModel(out_channels=32)


@pytest.fixture
def backbone_model_fixture(request: FixtureRequest) -> BackboneModel:
    model = request.getfixturevalue("model_fixture")
    config = BackboneModelConfig(
        type="DummyModel",
        input_channels_order="chw",
        in_channels=32,
        out_channels=16,
        pretrained=False,
    )
    return BackboneModel(model=model, config=config)


@pytest.fixture
def fusion_model_fixture() -> FusionModel:
    backbone_config = BackboneModelConfig(
        type="test", input_channels_order="chw", in_channels=3, out_channels=16, pretrained=False
    )
    return FusionModel(
        fusion_config=FusionModelConfig(out_channels=32),
        backbone_2d_config=backbone_config,
        backbone_3d_config=backbone_config,
    )


@pytest.fixture
def segmentation_model_fixture(request: FixtureRequest) -> SegmentationModel:
    segmentation_config = SegmentationModelConfig(out_channels=8)
    return SegmentationModel(
        segmentation_config=segmentation_config,
        backbone2d=request.getfixturevalue("backbone_model_fixture"),
        backbone3d=request.getfixturevalue("backbone_model_fixture"),
        fusion=request.getfixturevalue("fusion_model_fixture"),
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
