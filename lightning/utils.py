"""PyTorch Lightning Module for segmentation."""

from typing import Iterator

from torch import nn, optim

from config.config_schema import LossConfig, OptimizerConfig, SchedulerConfig


def instantiate_loss(config: LossConfig) -> nn.Module:
    """Return a loss function instantiated from a configuration."""
    loss_fn_cls = getattr(nn, config.type)
    loss_fn_args = config.keep_valid_args(loss_fn_cls)
    return loss_fn_cls(**loss_fn_args)  # type: ignore


def instantiate_optimizer(
    config: OptimizerConfig, parameters: Iterator[nn.Parameter]
) -> optim.Optimizer:
    """Return an optimizer instantiated from a configuration."""
    optimizer_cls = getattr(optim, config.type)
    optimizer_args = config.keep_valid_args(optimizer_cls)
    return optimizer_cls(parameters, **optimizer_args)  # type: ignore


def instantiate_scheduler(
    config: SchedulerConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler._LRScheduler:
    """Return a learning rate scheduler instantiated from a configuration."""
    scheduler_cls = getattr(optim.lr_scheduler, config.type)
    scheduler_args = config.keep_valid_args(scheduler_cls)
    return scheduler_cls(optimizer, **scheduler_args)  # type: ignore
