#!/usr/bin/env python3
"""Sweep the privacy budget and hypervector size.

Reproduces the shape of Figure 10 in the paper: accuracy rises with the
hypervector size and falls as the privacy budget tightens.

Usage
-----
    python scripts/sweep_privacy_budget.py \
        --dataset uci_har --epsilons 4 6 8 10 --dimensions 1000 5000 10000

Results are written to ``results/sweep-<timestamp>.json`` and, if matplotlib is
installed, a heatmap is saved alongside it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

from fedhdprivacy import ExperimentConfig, load_dataset, run_federated_training
from fedhdprivacy.data import AVAILABLE_DATASETS
from fedhdprivacy.utils import configure_logging, timestamp

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="uci_har")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    dataset = load_dataset(args.dataset, n_clients=args.clients, seed=args.seed)
    LOGGER.info("%s", dataset.summary())

    grid = list(itertools.product(args.dimensions, args.epsilons))
    records = []

    for index, (dimensions, epsilon) in enumerate(grid, start=1):
        LOGGER.info("[%d/%d] D=%d  eps=%g", index, len(grid), dimensions, epsilon)
        config = ExperimentConfig(
            dataset=args.dataset,
            n_clients=args.clients,
            rounds=args.rounds,
            dimensions=dimensions,
            epsilon=epsilon,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        history = run_federated_training(dataset, config, progress=False)
        records.append(
            {
                "dimensions": dimensions,
                "epsilon": epsilon,
                "accuracy": history.final_report.accuracy,
                "macro_fpr": history.final_report.macro_fpr,
                "macro_fnr": history.final_report.macro_fnr,
                "accuracy_per_round": history.accuracies,
                "seconds": history.total_seconds,
            }
        )
        LOGGER.info("    accuracy = %.4f", records[-1]["accuracy"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sweep-{timestamp()}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps({"dataset": args.dataset, "results": records}, indent=2)
    )
    LOGGER.info("Wrote %s", output_dir / f"{stem}.json")

    _maybe_plot(records, args, output_dir / f"{stem}.png")
    return 0


def _maybe_plot(records, args, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        LOGGER.info("matplotlib not installed; skipping the heatmap.")
        return

    grid = np.zeros((len(args.dimensions), len(args.epsilons)))
    for record in records:
        row = args.dimensions.index(record["dimensions"])
        col = args.epsilons.index(record["epsilon"])
        grid[row, col] = record["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(1.4 * len(args.epsilons) + 2, 0.8 * len(args.dimensions) + 2))
    image = ax.imshow(grid, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(args.epsilons)), [f"{e:g}" for e in args.epsilons])
    ax.set_yticks(range(len(args.dimensions)), [f"{d:,}" for d in args.dimensions])
    ax.set_xlabel("Privacy budget  $\\epsilon$")
    ax.set_ylabel("Hypervector size  $D$")
    ax.set_title(f"FedHDPrivacy accuracy (%) - {args.dataset}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
