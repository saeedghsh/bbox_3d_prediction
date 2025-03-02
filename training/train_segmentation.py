"""Training script for segmentation using PyTorch Lightning."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config.config_schema import (
    BackboneConfig,
    DataConfig,
    LayerConfig,
    LoggingConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from config.configuration import read_config
from lightning.lightning_data_module import MultimodalDataModule
from lightning.segmentation_lightning_module import SegmentationLightningModule
from models.factories import build_predictor_model

PIPELINE_NAME = "segmentation"


def _backbone_channels_order(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        branch_name: branch_config["input_channels_order"]
        for branch_name, branch_config in pipeline_config["models"]["branches"].items()
    }


def _parse_head_layers(head_layers: Optional[List[Dict[str, Any]]]) -> List[LayerConfig]:
    return [LayerConfig.from_dict(layer) for layer in head_layers] if head_layers else []


def _parse_pipeline_config(pipeline_config: dict) -> Dict[str, Any]:
    for branch_config in pipeline_config["models"]["branches"].values():
        if branch_config["backbone"] is not None:
            branch_config["backbone"] = BackboneConfig(**branch_config["backbone"])
        branch_config["head_layers"] = _parse_head_layers(branch_config["head_layers"])
    pipeline_config["models"]["fusion"]["head_layers"] = _parse_head_layers(
        pipeline_config["models"]["fusion"]["head_layers"]
    )
    pipeline_config["models"]["predictor"]["head_layers"] = _parse_head_layers(
        pipeline_config["models"]["predictor"]["head_layers"]
    )
    pipeline_config["training"] = TrainingConfig.from_dict(pipeline_config["training"])
    pipeline_config["loss"] = LossConfig.from_dict(pipeline_config["loss"])
    pipeline_config["optimizer"] = OptimizerConfig.from_dict(pipeline_config["optimizer"])
    pipeline_config["scheduler"] = SchedulerConfig.from_dict(pipeline_config["scheduler"])
    return pipeline_config


def main(_: Sequence[str]) -> int:
    # pylint: disable=missing-function-docstring, too-many-locals, unused-variable
    data_config_path = Path("./config/data.yaml")
    log_config_path = Path("./config/logging.yaml")
    pipeline_config_path = Path("./config/segmentation_pipeline.yaml")

    # Read, parse, and prepare configuration files
    data_config = DataConfig.from_dict(read_config(data_config_path))
    log_config = LoggingConfig.from_dict(read_config(log_config_path))
    pipeline_config = read_config(pipeline_config_path)
    pipeline_config = _parse_pipeline_config(pipeline_config)

    # Build the model
    predictor_model = build_predictor_model(pipeline_config, data_config)

    # Set up LightningModule and DataModule
    lightning_module = SegmentationLightningModule(
        model=predictor_model,
        loss_config=pipeline_config["loss"],
        optimizer_config=pipeline_config["optimizer"],
        scheduler_config=pipeline_config["scheduler"],
    )
    data_module = MultimodalDataModule(
        data_dir=data_config.dataset_dir,
        split_test=data_config.split_test,
        split_val=data_config.split_val,
        batch_size=pipeline_config["training"].batch_size,
        data_channels_order=data_config.channels_order,
        backbone_channels_order=_backbone_channels_order(pipeline_config),
    )

    # Set up callbacks and trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=log_config.checkpoint_dir,
        filename="{PIPELINE_NAME}-{epoch:02d}-{val_loss:.2f}",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    logger = TensorBoardLogger(save_dir=log_config.log_dir, name=f"{PIPELINE_NAME}_logs")
    trainer = pl.Trainer(
        max_epochs=pipeline_config["training"].epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(lightning_module, datamodule=data_module)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
