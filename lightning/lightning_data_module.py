"""LightningDataModule handling train/validation/test splits and data loading."""

from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split

from dataset_handler.dataset_handler import DatasetHandler, build_transform


class MultimodalDataModule(pl.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """DataModule for multimodal data."""

    def __init__(
        self,
        data_dir: Path,
        split_test: float,
        split_val: float,
        batch_size: int,
        data_channels_order: Dict[str, str],
        backbone_channels_order: Dict[str, str],
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self._data_dir = data_dir
        self._split_test = split_test
        self._split_val = split_val
        self._batch_size = batch_size
        self._frame_transform = build_transform(
            rgb_source_order=data_channels_order["rgb"],
            pc_source_order=data_channels_order["pc"],
            rgb_target_order=backbone_channels_order["rgb"],
            pc_target_order=backbone_channels_order["pc"],
        )
        self._train_dataset: Subset
        self._val_dataset: Subset
        self._test_dataset: Subset

    def prepare_data(self) -> None:
        pass  # Data is assumed to be already downloaded/present.

    def setup(self, stage: Optional[str] = None) -> None:  # pylint: disable=unused-argument
        """Set up datasets and define transforms based on model input requirements."""
        full_dataset = DatasetHandler(data_dir=self._data_dir, transform=self._frame_transform)
        dataset_len = len(full_dataset)
        test_size = int(dataset_len * self._split_test)
        val_size = int(dataset_len * self._split_val)
        train_size = dataset_len - val_size - test_size
        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

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
