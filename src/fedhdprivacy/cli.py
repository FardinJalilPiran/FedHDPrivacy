"""Command-line entry point.

Examples
--------
    fedhdprivacy --dataset uci_har --clients 8 --rounds 10 --epsilon 10
    fedhdprivacy --config configs/default.yaml --epsilon 4
    fedhdprivacy --dataset synthetic --dimensions 2000 --rounds 3 --no-dp
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ExperimentConfig
from .data import AVAILABLE_DATASETS, load_dataset
from .federated import run_federated_training
from .hdc import AVAILABLE_ENCODERS
from .utils import configure_logging, save_history, timestamp

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedhdprivacy",
        description="Privacy-preserving federated learning with hyperdimensional computing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file to start from")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=AVAILABLE_DATASETS, default=None)
    data.add_argument(
        "--partition",
        choices=["natural", "iid", "dirichlet"],
        default=None,
        help="how to split the data across clients",
    )
    data.add_argument("--dirichlet-alpha", type=float, default=None)
    data.add_argument("--data-root", type=str, default=None, help="download cache directory")

    federation = parser.add_argument_group("federation")
    federation.add_argument("--clients", type=int, default=None, dest="n_clients")
    federation.add_argument("--rounds", type=int, default=None)
    federation.add_argument("--local-epochs", type=int, default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--dimensions", type=int, default=None, help="hypervector size D")
    model.add_argument(
        "--encoder",
        choices=list(AVAILABLE_ENCODERS),
        default=None,
        help="projection is fast and memory-light; density matches the original notebook",
    )

    privacy = parser.add_argument_group("privacy")
    privacy.add_argument("--epsilon", type=float, default=None, help="privacy budget")
    privacy.add_argument(
        "--no-dp",
        action="store_true",
        help="disable differential privacy (non-private accuracy ceiling)",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=None)
    runtime.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    runtime.add_argument("--output-dir", type=str, default=None)
    runtime.add_argument("--run-name", type=str, default=None)
    runtime.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Start from a YAML file (or defaults) and apply any explicit CLI flags."""
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    overridable = [
        "dataset",
        "partition",
        "dirichlet_alpha",
        "data_root",
        "n_clients",
        "rounds",
        "local_epochs",
        "dimensions",
        "encoder",
        "epsilon",
        "seed",
        "device",
        "output_dir",
        "run_name",
    ]
    for name in overridable:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)

    if args.no_dp:
        config.differential_privacy = False

    config.__post_init__()  # re-validate after the overrides
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.WARNING if args.quiet else logging.INFO)

    config = config_from_args(args)
    LOGGER.info("Configuration: %s", config.describe())

    dataset = load_dataset(
        name=config.dataset,
        n_clients=config.n_clients,
        partition=config.partition,
        root=config.data_root,
        seed=config.seed,
        dirichlet_alpha=config.dirichlet_alpha,
    )

    history = run_federated_training(dataset, config, progress=not args.quiet)

    run_name = config.run_name or f"{config.dataset}-eps{config.epsilon:g}-{timestamp()}"
    path = save_history(history, config.output_dir, run_name)

    print()
    print("Final global model")
    print("------------------")
    print(f"  {history.final_report}")
    print(f"  wall clock: {history.total_seconds:.1f}s")
    print(f"  results:    {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
