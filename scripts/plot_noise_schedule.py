#!/usr/bin/env python3
"""Plot required vs. injected noise across communication rounds.

Reproduces Figure 8b of the paper.  No training is involved - the schedule is
closed-form - so this runs in milliseconds and is a good way to see what the
adaptive mechanism actually buys you before committing to a full experiment.

Usage
-----
    python scripts/plot_noise_schedule.py --rounds 50 --clients 8 --samples 1000
"""

from __future__ import annotations

import argparse

from fedhdprivacy.privacy import (
    additional_noise_variance,
    cumulative_noise_variance,
    required_noise_variance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1000, help="samples per client per round L")
    parser.add_argument("--dimensions", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="results/noise_schedule.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = dict(
        dimensions=args.dimensions,
        epsilon=args.epsilon,
        n_clients=args.clients,
        samples_per_round=args.samples,
    )

    rounds = list(range(1, args.rounds + 1))
    required = [required_noise_variance(**kwargs, round_index=r) for r in rounds]
    cumulative = [cumulative_noise_variance(**kwargs, round_index=r) for r in rounds]
    added = [additional_noise_variance(**kwargs, round_index=r) for r in rounds]

    print(f"{'round':>6} {'required':>14} {'cumulative':>14} {'added':>14} {'saved':>8}")
    for r, req, cum, add in zip(rounds, required, cumulative, added):
        saved = 100.0 * (1.0 - add / req) if req else 0.0
        print(f"{r:>6} {req:>14.2f} {cum:>14.2f} {add:>14.2f} {saved:>7.1f}%")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping the plot.")
        return 0

    from pathlib import Path

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, required, marker="s", markevery=5, label="required  $\\xi^r_k$")
    ax.plot(rounds, added, marker="o", markevery=5, linestyle="--", label="added  $\\Gamma^r_k$")
    ax.plot(
        rounds, cumulative, marker="^", markevery=5, alpha=0.7, label="inherited  $\\Psi^{r-1}_k$"
    )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Noise variance")
    ax.set_title(
        f"Adaptive noise schedule  (K={args.clients}, L={args.samples}, $\\epsilon$={args.epsilon:g})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
