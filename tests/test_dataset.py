# pylint: disable=missing-module-docstring, missing-function-docstring

import numpy as np
import pytest
import torch

from data.data_structure import Frame


@pytest.fixture
def numpy_frame() -> Frame:
    return Frame(
        rgb=np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8),
        pc=np.random.rand(3, 100, 200).astype(np.float32),
        mask=np.random.randint(0, 10, (5, 100, 200), dtype=np.int64),
        bbox3d=np.random.rand(5, 8, 3).astype(np.float32),
    )


@pytest.fixture
def tensor_frame(request: pytest.FixtureRequest) -> Frame:
    frame_as_numpy = request.getfixturevalue("numpy_frame")
    return Frame.as_tensor(frame_as_numpy)


def test_frame_type_consistency(request: pytest.FixtureRequest) -> None:
    frame_as_numpy = request.getfixturevalue("numpy_frame")
    assert isinstance(frame_as_numpy.rgb, np.ndarray)
    assert isinstance(frame_as_numpy.pc, np.ndarray)
    assert isinstance(frame_as_numpy.mask, np.ndarray)
    assert isinstance(frame_as_numpy.bbox3d, np.ndarray)


def test_frame_inconsistent_types_raises_error() -> None:
    with pytest.raises(ValueError, match="inconsistent member types"):
        Frame(
            rgb=np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8),
            pc=torch.rand(3, 100, 200),  # Inconsistent type
            mask=np.random.randint(0, 10, (5, 100, 200), dtype=np.int64),
            bbox3d=np.random.rand(5, 8, 3).astype(np.float32),
        )


def test_frame_as_tensor(request: pytest.FixtureRequest) -> None:
    # from numpy to tensor
    frame_as_numpy = request.getfixturevalue("numpy_frame")
    frame_as_tensor = Frame.as_tensor(frame_as_numpy)
    assert isinstance(frame_as_tensor.rgb, torch.Tensor)
    assert isinstance(frame_as_tensor.pc, torch.Tensor)
    assert isinstance(frame_as_tensor.mask, torch.Tensor)
    assert isinstance(frame_as_tensor.bbox3d, torch.Tensor)
    # already tensor, move to target device
    frame_as_tensor = request.getfixturevalue("tensor_frame")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_on_device = Frame.as_tensor(frame_as_tensor, device=device)
    assert frame_on_device.rgb.device == device  # type: ignore[union-attr]
    assert frame_on_device.pc.device == device  # type: ignore[union-attr]
    assert frame_on_device.mask.device == device  # type: ignore[union-attr]
    assert frame_on_device.bbox3d.device == device  # type: ignore[union-attr]
    # already tensor, already on target device
    frame_as_tensor = request.getfixturevalue("tensor_frame")
    same_frame = Frame.as_tensor(frame_as_tensor)  # No device change
    assert same_frame is frame_as_tensor


def test_frame_as_numpy(request: pytest.FixtureRequest) -> None:
    # from tensor to numpy
    frame_as_tensor = request.getfixturevalue("tensor_frame")
    frame_as_numpy = Frame.as_numpy(frame_as_tensor)
    assert isinstance(frame_as_numpy.rgb, np.ndarray)
    assert isinstance(frame_as_numpy.pc, np.ndarray)
    assert isinstance(frame_as_numpy.mask, np.ndarray)
    assert isinstance(frame_as_numpy.bbox3d, np.ndarray)
    # already numpy
    frame_as_numpy = request.getfixturevalue("numpy_frame")
    frame_as_numpy = Frame.as_numpy(frame_as_numpy)
    assert isinstance(frame_as_numpy.rgb, np.ndarray)
    assert isinstance(frame_as_numpy.pc, np.ndarray)
    assert isinstance(frame_as_numpy.mask, np.ndarray)
    assert isinstance(frame_as_numpy.bbox3d, np.ndarray)
