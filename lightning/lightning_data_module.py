"""LightningDataModule handling train/validation/test splits and data loading."""

from typing import Callable, Dict, Optional, cast

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split

from config.config_schema import BackboneModelConfig, DataConfig, TrainingConfig
from dataset_handler.dataset_handler import DatasetHandler, Frame


def _reorder_channels(data: np.ndarray, source_order: str, target_order: str) -> np.ndarray:
    """Reorders the axes of the input data from source_order to target_order.

    source_order: The current axis order as a string (e.g., "hwc", "chw").
    target_order: The desired axis order as a string (e.g., "chw", "hwc").
    """
    if set(source_order) != set(target_order):
        raise ValueError(f"Incompatible source and target orders: {source_order} -> {target_order}")

    permutation = tuple(source_order.index(axis) for axis in target_order)
    return np.transpose(data, permutation)


def _dtype_torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Converts a torch dtype to a numpy dtype."""
    return cast(np.dtype, torch.empty(0, dtype=dtype).numpy().dtype)


class MultimodalDataModule(pl.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """DataModule for multimodal data."""

    def __init__(
        self,
        data_config: DataConfig,
        training_config: TrainingConfig,
        backbone_2d_config: BackboneModelConfig,
        backbone_3d_config: BackboneModelConfig,
        input_branches_dtype: Dict[str, torch.dtype],
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self._data_config = data_config
        self._training_config = training_config
        self._backbone_2d_config = backbone_2d_config
        self._backbone_3d_config = backbone_3d_config
        self._batch_size = training_config.batch_size
        self._input_branches_dtype = input_branches_dtype
        self._train_dataset: Subset
        self._val_dataset: Subset
        self._test_dataset: Subset

    def prepare_data(self) -> None:
        pass  # Data is assumed to be already downloaded/present.

    def setup(self, stage: Optional[str] = None) -> None:  # pylint: disable=unused-argument
        """Set up datasets and define transforms based on model input requirements."""
        full_dataset = DatasetHandler(config=self._data_config, transform=self._build_transform())

        dataset_len = len(full_dataset)
        test_size = int(dataset_len * self._data_config.split_test)
        val_size = int(dataset_len * self._data_config.split_val)
        train_size = dataset_len - val_size - test_size

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def _build_transform(self) -> Callable[[Frame], Frame]:
        """Builds a transform function to reorder input shapes based on model requirements."""
        rgb_source_order = self._data_config.rgb_channels_order
        pc_source_order = self._data_config.pc_channels_order
        rgb_target_order = self._backbone_2d_config.input_channels_order
        pc_target_order = self._backbone_3d_config.input_channels_order

        rgb_dtype = _dtype_torch_to_numpy(self._input_branches_dtype["2d"])
        pc_dtype = _dtype_torch_to_numpy(self._input_branches_dtype["3d"])

        def transform(frame: Frame) -> Frame:
            frame.rgb = _reorder_channels(frame.rgb, rgb_source_order, rgb_target_order)
            frame.rgb = frame.rgb.astype(rgb_dtype)
            frame.pc = _reorder_channels(frame.pc, pc_source_order, pc_target_order)
            frame.pc = frame.pc.astype(pc_dtype)
            return frame

        return transform

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4
        )
