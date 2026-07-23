"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["ExperimentConfig"]


@dataclass
class ExperimentConfig:
    """Every knob for one FedHDPrivacy run.

    Attributes
    ----------
    dimensions:
        Hypervector size ``D``.  Larger is more accurate but costs time, memory
        and energy roughly linearly (Figure 11 of the paper).
    epsilon:
        Privacy budget.  Lower means stronger privacy and more injected noise.
    rounds:
        Number of communication rounds ``R``.
    n_clients:
        Number of clients ``K``.
    local_epochs:
        Maximum retraining passes a client makes over its new shard each round.
    encoder:
        ``"projection"`` (fast, memory-light) or ``"density"`` (torchhd, matches
        the original notebook).  See :mod:`fedhdprivacy.hdc`.
    differential_privacy:
        Set to ``False`` for the no-privacy upper bound.
    """

    # data
    dataset: str = "uci_har"
    partition: str = "natural"
    dirichlet_alpha: float = 0.5
    data_root: str = "data"

    # federation
    n_clients: int = 8
    rounds: int = 10
    local_epochs: int = 10

    # model
    dimensions: int = 10000
    encoder: str = "projection"  # "projection" | "density"

    # privacy
    differential_privacy: bool = True
    epsilon: float = 10.0

    # runtime
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    output_dir: str = "results"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if self.differential_privacy and self.epsilon <= 0:
            raise ValueError("epsilon must be > 0 when differential privacy is enabled")
        if self.encoder not in ("projection", "density"):
            raise ValueError(f"encoder must be 'projection' or 'density', got {self.encoder!r}")

    # ------------------------------------------------------------------ io

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        import yaml  # noqa: PLC0415

        with Path(path).open() as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> ExperimentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def describe(self) -> str:
        privacy = f"eps={self.epsilon}" if self.differential_privacy else "DP disabled"
        return (
            f"{self.dataset} | K={self.n_clients} | R={self.rounds} | "
            f"D={self.dimensions} | {self.encoder} | {privacy} | seed={self.seed}"
        )
