from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class LocalPaths:
    raw: str = MISSING
    processed: str = MISSING
    final: str = MISSING


@dataclass
class MainParams:
    timeout: int
    fpr_max: float
    metric: str
    seed: int


@dataclass
class LGBMParams:
    model_name: str
    objective: str
    metric: str
    verbosity: int
    boosting_type: str
    lambda_l1: float
    lambda_l2: float
    num_leaves: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    min_child_samples: int
    random_state: int


@dataclass
class ModelParams:
    size: dict[str, dict]


@dataclass
class MainConfig:
    main_params: MainParams
    model_params: ModelParams
    local_paths: LocalPaths = field(default_factory=LocalPaths)
