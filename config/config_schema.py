"""Dataclasses for configuration schema."""

# pylint: disable=missing-class-docstring

import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple, Type, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BackboneConfig:  # pylint: disable=missing-class-docstring
    model_name: str = "ResNet18"
    remove_head: bool = True
    freeze_backbone: bool = True
    pretrained: bool = True


@dataclass
class LayerConfig:  # pylint: disable=missing-class-docstring
    type: Literal["Conv2d", "ReLU", "ConvTranspose2d"] = "ReLU"
    kwargs: Dict[str, Any] = field(default_factory=dict)  # only what each layer needs/accepts

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "LayerConfig":
        # pylint: disable=missing-function-docstring
        config_copy = dict(config)
        layer_type = config_copy.pop("type")
        return LayerConfig(type=layer_type, kwargs=config_copy)


@dataclass
class BaseConfig:
    """Base class for configuration dataclasses."""

    @classmethod
    def from_dict(cls: Type[T], config: Dict[str, Any]) -> T:
        """Dynamically create an instance from a dictionary."""
        return cls(**config)


@dataclass
class DataConfig(BaseConfig):
    dataset_dir: Path
    split_val: float
    split_test: float
    target_height: int
    target_width: int
    rgb_channels_order: str
    pc_channels_order: str

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir).resolve()
        self.split_val = float(self.split_val)
        self.split_test = float(self.split_test)
        self.target_height = int(self.target_height)
        self.target_width = int(self.target_width)
        self.rgb_channels_order = self.rgb_channels_order.lower()
        self.pc_channels_order = self.pc_channels_order.lower()


@dataclass
class LoggingConfig(BaseConfig):
    log_dir: Path
    checkpoint_dir: Path

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir).resolve()
        self.checkpoint_dir = Path(self.checkpoint_dir).resolve()


@dataclass
class TrainingConfig(BaseConfig):
    batch_size: int
    epochs: int

    def __post_init__(self) -> None:
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)


@dataclass
class ValidArgsMixin:
    """CAUTION: This method assumes that all config parameter names in the YAML
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

    def keep_valid_args(self, cls: Callable[..., Any]) -> Dict[str, Any]:
        """Keep only valid arguments for the given class."""
        valid_params = inspect.signature(cls).parameters
        args = {k: v for k, v in asdict(self).items() if k in valid_params}
        return args


@dataclass
class LossConfig(BaseConfig, ValidArgsMixin):
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
class OptimizerConfig(BaseConfig, ValidArgsMixin):
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

    def __post_init__(self) -> None:
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.betas = tuple(float(beta) for beta in self.betas)  # type: ignore[assignment]
        self.eps = float(self.eps)


@dataclass
class SchedulerConfig(BaseConfig, ValidArgsMixin):
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

    def __post_init__(self) -> None:
        self.step_size = int(self.step_size)
        self.gamma = float(self.gamma)
