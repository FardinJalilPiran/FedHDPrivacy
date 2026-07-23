"""FedHDPrivacy: privacy-preserving federated learning with hyperdimensional computing.

Reference implementation of

    F. Jalil Piran, Z. Chen, M. Imani, F. Imani,
    "Privacy-Preserving Federated Learning with Differentially Private
    Hyperdimensional Computing", Computers and Electrical Engineering,
    123:110261, 2025.  https://doi.org/10.1016/j.compeleceng.2025.110261

Quick start
-----------
>>> from fedhdprivacy import ExperimentConfig, load_dataset, run_federated_training
>>> config = ExperimentConfig(dataset="synthetic", rounds=3, dimensions=2000)
>>> data = load_dataset("synthetic", n_clients=config.n_clients)
>>> history = run_federated_training(data, config)
"""

from .config import ExperimentConfig
from .data import AVAILABLE_DATASETS, ClientData, FederatedDataset, load_dataset
from .federated import RoundResult, TrainingHistory, noise_schedule, run_federated_training
from .hdc import AVAILABLE_ENCODERS, Encoder, HDClassifier, build_encoder
from .metrics import ClassificationReport, evaluate
from .privacy import (
    GaussianNoiseMechanism,
    additional_noise_variance,
    cumulative_noise_variance,
    required_noise_variance,
)

__version__ = "1.0.1"

__all__ = [
    "AVAILABLE_DATASETS",
    "AVAILABLE_ENCODERS",
    "ClassificationReport",
    "ClientData",
    "Encoder",
    "ExperimentConfig",
    "FederatedDataset",
    "GaussianNoiseMechanism",
    "HDClassifier",
    "RoundResult",
    "TrainingHistory",
    "__version__",
    "additional_noise_variance",
    "build_encoder",
    "cumulative_noise_variance",
    "evaluate",
    "load_dataset",
    "noise_schedule",
    "required_noise_variance",
    "run_federated_training",
]
