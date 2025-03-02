"""PyTorch Lightning Module for segmentation."""

from typing import Any, Dict, Tuple, cast

import pytorch_lightning as pl
from torch import Tensor

from config.config_schema import LossConfig, OptimizerConfig, SchedulerConfig
from lightning.utils import instantiate_loss, instantiate_optimizer, instantiate_scheduler
from models.predictor import Predictor


class SegmentationLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for segmentation.

    Wraps a Predictor, defines the loss function, and implements the
    training, validation, and test steps.
    """

    # pylint: disable=unused-argument, arguments-differ

    BatchType = Tuple[Tensor, Tensor, Tensor, Tensor]

    def __init__(
        self,
        model: Predictor,
        loss_config: LossConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._loss_fn = instantiate_loss(loss_config)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass through the segmentation model."""
        return cast(Tensor, self._model(*args, **kwargs))

    def _step(self, batch: BatchType, step_name: str) -> Tensor:
        image, pointcloud, mask, _ = batch  # bbox3d is ignored for segmentation
        logits = self.forward([image, pointcloud])
        loss: Tensor = self._loss_fn(logits, mask)
        self.log(f"{step_name}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        return self._step(batch, "validation")

    def test_step(self, batch: BatchType, batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        optimizer = instantiate_optimizer(self._optimizer_config, self.parameters())
        scheduler = instantiate_scheduler(self._scheduler_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
