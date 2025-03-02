"""PyTorch Lightning Module for segmentation."""

from typing import Any, Dict, Tuple, cast

import pytorch_lightning as pl
from torch import Tensor

from config.config_schema import LossConfig, OptimizerConfig, SchedulerConfig, TrainingConfig
from lightning.utils import instantiate_loss, instantiate_optimizer, instantiate_scheduler
from models.predictor import Predictor


class SegmentationLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for segmentation.

    Wraps a Predictor, defines the loss function, and implements the
    training, validation, and test steps.
    """

    BatchType = Tuple[Tensor, Tensor, Tensor, Tensor]

    def __init__(
        self,
        model: Predictor,
        loss_config: LossConfig,
        training_config: TrainingConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        super().__init__()
        self._model = model
        self._training_config = training_config
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._loss_fn = instantiate_loss(loss_config)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass through the segmentation model."""
        return cast(Tensor, self._model(*args, **kwargs))

    def training_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        # pylint: disable=unused-argument, arguments-differ
        image, pointcloud, mask, _ = batch  # bbox3d is ignored for segmentation
        logits = self.forward([image, pointcloud])
        loss: Tensor = self._loss_fn(logits, mask)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        # pylint: disable=unused-argument, arguments-differ
        image, pointcloud, mask, _ = batch
        logits = self.forward([image, pointcloud])
        loss: Tensor = self._loss_fn(logits, mask)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        # pylint: disable=unused-argument, arguments-differ
        image, pointcloud, mask, _ = batch
        logits = self.forward([image, pointcloud])
        loss: Tensor = self._loss_fn(logits, mask)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        optimizer = instantiate_optimizer(self._optimizer_config, self.parameters())
        scheduler = instantiate_scheduler(self._scheduler_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
