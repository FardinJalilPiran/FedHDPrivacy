"""The FedHDPrivacy training loop.

One round proceeds as follows:

1. Every client receives the current global model and a fresh shard of ``L``
   samples - the continual-learning regime the paper targets, where data keeps
   arriving instead of being available up front.
2. Round 1 bundles the shard into class hypervectors from scratch; later rounds
   retrain the downloaded global model on the new shard.
3. Each client perturbs its model with Gaussian noise whose variance is the
   *difference* between what privacy requires now and what the downloaded model
   already carries.
4. The server averages the noisy local models.  It adds nothing of its own:
   Theorems 3 and 5 show the aggregate is already private, because averaging
   ``K`` independently noised models divides the variance by ``K`` more slowly
   than the sensitivity of the aggregate shrinks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import ExperimentConfig
from .data import FederatedDataset
from .hdc import HDClassifier, average, build_encoder
from .metrics import ClassificationReport, evaluate
from .privacy import GaussianNoiseMechanism
from .utils import resolve_device, set_seed

LOGGER = logging.getLogger(__name__)

__all__ = ["RoundResult", "TrainingHistory", "run_federated_training"]


@dataclass
class RoundResult:
    round_index: int
    accuracy: float
    macro_fpr: float
    macro_fnr: float
    noise_required: float
    noise_cumulative: float
    noise_added: float
    seconds: float

    def to_dict(self) -> dict:
        return {
            "round": self.round_index,
            "accuracy": self.accuracy,
            "macro_fpr": self.macro_fpr,
            "macro_fnr": self.macro_fnr,
            "noise_variance_required": self.noise_required,
            "noise_variance_cumulative": self.noise_cumulative,
            "noise_variance_added": self.noise_added,
            "seconds": self.seconds,
        }


@dataclass
class TrainingHistory:
    config: ExperimentConfig
    rounds: list[RoundResult] = field(default_factory=list)
    final_report: ClassificationReport | None = None
    dataset_summary: str = ""
    samples_per_round: int = 0
    total_seconds: float = 0.0

    @property
    def accuracies(self) -> list[float]:
        return [r.accuracy for r in self.rounds]

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "dataset_summary": self.dataset_summary,
            "samples_per_round": self.samples_per_round,
            "total_seconds": self.total_seconds,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_report": self.final_report.to_dict() if self.final_report else None,
        }


def run_federated_training(
    dataset: FederatedDataset,
    config: ExperimentConfig,
    progress: bool = True,
) -> TrainingHistory:
    """Run ``config.rounds`` communication rounds and return the full history."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    # Deliberately a CPU generator even when training on GPU: the sampling
    # helpers move draws to the target device, so a seeded run gives the same
    # noise and the same shuffle order on either backend.
    generator = torch.Generator().manual_seed(config.seed)

    n_classes = dataset.n_classes
    samples_per_round = dataset.min_client_samples() // config.rounds
    if samples_per_round < 1:
        raise ValueError(
            f"Not enough data: the smallest client holds {dataset.min_client_samples()} "
            f"samples but {config.rounds} rounds were requested."
        )

    LOGGER.info("Device: %s", device)
    LOGGER.info("%s", dataset.summary())
    LOGGER.info("Samples per client per round (L): %d", samples_per_round)

    encoder = build_encoder(
        n_features=dataset.n_features,
        dimensions=config.dimensions,
        backend=config.encoder,
        device=device,
        seed=config.seed,
    )
    mechanism = GaussianNoiseMechanism(
        dimensions=config.dimensions,
        epsilon=config.epsilon,
        n_clients=dataset.n_clients,
        samples_per_round=samples_per_round,
        enabled=config.differential_privacy,
    )

    # Encoding the test set once keeps evaluation off the critical path.
    test_encoded = encoder.encode(torch.from_numpy(dataset.test_x))
    test_labels = torch.from_numpy(dataset.test_y).long()

    history = TrainingHistory(
        config=config,
        dataset_summary=dataset.summary(),
        samples_per_round=samples_per_round,
    )
    global_model: HDClassifier | None = None
    started = time.perf_counter()

    for round_index in range(1, config.rounds + 1):
        round_started = time.perf_counter()
        local_models: list[HDClassifier] = []

        for client in dataset.clients:
            start = (round_index - 1) * samples_per_round
            shard_x = client.x[start : start + samples_per_round]
            shard_y = client.y[start : start + samples_per_round]

            encoded = encoder.encode(torch.from_numpy(shard_x))
            labels = torch.from_numpy(shard_y).long().to(device)

            if global_model is None:
                model = HDClassifier.zeros(n_classes, config.dimensions, device=device)
                model.bundle(encoded, labels)
            else:
                model = global_model.clone()
                model.retrain(encoded, labels, n_epochs=config.local_epochs, generator=generator)

            noisy = mechanism.perturb(model.class_hypervectors, round_index, generator=generator)
            local_models.append(HDClassifier(noisy))

        global_model = average(local_models)

        predictions = global_model.predict(test_encoded).cpu().numpy()
        report = evaluate(test_labels.numpy(), predictions, n_classes)
        variances = mechanism.variances(round_index)

        result = RoundResult(
            round_index=round_index,
            accuracy=report.accuracy,
            macro_fpr=report.macro_fpr,
            macro_fnr=report.macro_fnr,
            noise_required=variances["required"],
            noise_cumulative=variances["cumulative"],
            noise_added=variances["added"],
            seconds=time.perf_counter() - round_started,
        )
        history.rounds.append(result)
        history.final_report = report

        if progress:
            saved = (
                100.0 * (1.0 - variances["added"] / variances["required"])
                if variances["required"] > 0
                else 0.0
            )
            LOGGER.info(
                "round %2d/%d  acc=%.4f  FPR=%.4f  FNR=%.4f  noise saved=%5.1f%%  (%.1fs)",
                round_index,
                config.rounds,
                report.accuracy,
                report.macro_fpr,
                report.macro_fnr,
                saved,
                result.seconds,
            )

    history.total_seconds = time.perf_counter() - started
    return history


def noise_schedule(config: ExperimentConfig, samples_per_round: int) -> dict[str, np.ndarray]:
    """Required / cumulative / added noise variance per round, without training.

    Handy for reproducing Figure 8b, which shows that the injected noise stays
    well below the required noise as rounds accumulate.
    """
    mechanism = GaussianNoiseMechanism(
        dimensions=config.dimensions,
        epsilon=config.epsilon,
        n_clients=config.n_clients,
        samples_per_round=samples_per_round,
        enabled=True,
    )
    rows = [mechanism.variances(r) for r in range(1, config.rounds + 1)]
    return {
        "round": np.arange(1, config.rounds + 1),
        "required": np.array([r["required"] for r in rows]),
        "cumulative": np.array([r["cumulative"] for r in rows]),
        "added": np.array([r["added"] for r in rows]),
    }
