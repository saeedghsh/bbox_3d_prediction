"""Dataclasses for configuration schema."""

# pylint: disable=missing-class-docstring

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Literal, Tuple, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class for configuration dataclasses."""

    def keep_valid_args(self, cls: Callable[..., Any]) -> Dict[str, Any]:
        """Keep only valid arguments for the given class.

        CAUTION: This method assumes that all config parameter names in the YAML
        file match exactly the argument names of the target class constructors
        they are passed to.

        For example, the learning rate must be spelled 'lr' in both the config
        file and the corresponding dataclass (e.g., OptimizerConfig). If a
        mismatch occurs, such as defining 'learning_rate' instead of 'lr', this
        method will incorrectly filter out the config entry as invalid.

        As a result, the intended value will not be passed to the constructor,
        which may lead to unexpected behavior—especially if the argument is
        optional and has a default in PyTorch (e.g., Adam optimizer's 'lr'
        default of 1e-3).
        """
        valid_params = inspect.signature(cls).parameters
        args = {k: v for k, v in asdict(self).items() if k in valid_params}
        return args

    @classmethod
    def from_dict(cls: Type[T], config: Dict[str, Any]) -> T:
        """Dynamically create an instance from a dictionary."""
        return cls(**config)


@dataclass
class BackboneModelConfig(BaseConfig):
    model_name: str
    in_channels: int
    out_features: int
    pretrained: bool


@dataclass
class TrainingConfig(BaseConfig):
    batch_size: int
    epochs: int


@dataclass
class LossConfig(BaseConfig):
    type: Literal[
        "BCELoss",
        "BCEWithLogitsLoss",
        "CrossEntropyLoss",
        "MSELoss",
        "L1Loss",
        "SmoothL1Loss",
        "HingeEmbeddingLoss",
        "HuberLoss",
        "KLDivLoss",
        "NLLLoss",
        "PoissonNLLLoss",
        "MultiMarginLoss",
        "MultiLabelSoftMarginLoss",
        "TripletMarginLoss",
    ]
    reduction: str


@dataclass
class OptimizerConfig(BaseConfig):
    type: Literal[
        "SGD",
        "Adam",
        "AdamW",
        "Adadelta",
        "Adagrad",
        "RMSprop",
        "Rprop",
        "NAdam",
        "RAdam",
        "ASGD",
        "LBFGS",
        "SparseAdam",
    ]
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float


@dataclass
class SchedulerConfig(BaseConfig):
    type: Literal[
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "ReduceLROnPlateau",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "CyclicLR",
        "OneCycleLR",
        "LambdaLR",
        "PolynomialLR",
    ]
    step_size: int
    gamma: float
